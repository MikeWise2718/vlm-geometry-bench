"""Tests for response parser module."""

import pytest
from vlm_geometry_bench.response_parser import (
    ResponseParser,
    ParsedCountResponse,
    ParsedLocateResponse,
    ParsedPatternResponse,
    ParsedSizeResponse,
    ParsedDefectResponse,
)


class TestParseCount:
    """Tests for COUNT response parsing."""

    @pytest.fixture
    def parser(self):
        return ResponseParser()

    def test_parse_just_number(self, parser):
        """Parse response that is just a number."""
        result = parser.parse_count("42")
        assert result.count == 42
        assert result.success

    def test_parse_number_with_unit(self, parser):
        """Parse number followed by 'spots'."""
        result = parser.parse_count("42 spots")
        assert result.count == 42

    def test_parse_natural_language_see(self, parser):
        """Parse 'I see N spots' format."""
        result = parser.parse_count("I can see 15 spots in the image")
        assert result.count == 15

    def test_parse_natural_language_there_are(self, parser):
        """Parse 'There are N' format."""
        result = parser.parse_count("There are 7 circles visible")
        assert result.count == 7

    def test_parse_answer_label(self, parser):
        """Parse 'Answer: N' format."""
        result = parser.parse_count("The answer is: 23")
        assert result.count == 23

    def test_parse_zero_no_spots(self, parser):
        """Parse response indicating zero spots."""
        result = parser.parse_count("I don't see any spots in this image")
        assert result.count == 0

    def test_parse_zero_empty(self, parser):
        """Parse response indicating empty image."""
        result = parser.parse_count("The image appears to be empty")
        assert result.count == 0

    def test_parse_empty_response(self, parser):
        """Empty response should fail."""
        result = parser.parse_count("")
        assert result.count is None
        assert not result.success
        assert result.parse_error

    def test_parse_no_number(self, parser):
        """Response without a number should fail."""
        result = parser.parse_count("Many spots are visible")
        assert result.count is None
        assert not result.success

    def test_parse_rejects_huge_numbers(self, parser):
        """Reject unreasonably large numbers."""
        result = parser.parse_count("I see 50000 spots")
        # Should fall back to finding any reasonable number or fail
        assert result.count is None or result.count <= 10000


class TestParseLocate:
    """Tests for LOCATE response parsing with normalized coordinates."""

    @pytest.fixture
    def parser(self):
        return ResponseParser()

    def test_parse_parentheses_format(self, parser):
        """Parse (x, y) format with normalized coordinates."""
        result = parser.parse_locate("(0.18, 0.67), (0.40, 0.29)")
        assert len(result.positions) == 2
        assert (0.18, 0.67) in result.positions
        assert (0.40, 0.29) in result.positions

    def test_parse_bracket_format(self, parser):
        """Parse [x, y] format with normalized coordinates."""
        result = parser.parse_locate("[[0.18, 0.67], [0.40, 0.29]]")
        assert len(result.positions) == 2

    def test_parse_one_per_line(self, parser):
        """Parse coordinates one per line."""
        result = parser.parse_locate("0.18, 0.67\n0.40, 0.29\n0.64, 0.82")
        assert len(result.positions) == 3

    def test_parse_floats(self, parser):
        """Parse floating point coordinates."""
        result = parser.parse_locate("(0.185, 0.672)")
        assert result.positions[0] == (0.185, 0.672)

    def test_parse_empty_indicates_no_spots(self, parser):
        """Response saying 'no spots' should return empty list."""
        result = parser.parse_locate("There are no spots in this image")
        assert result.positions == []
        assert result.success

    def test_parse_empty_response(self, parser):
        """Empty response should fail."""
        result = parser.parse_locate("")
        assert result.positions == []
        assert not result.success

    def test_rejects_out_of_bounds(self, parser):
        """Reject coordinates outside normalized range [0, 1]."""
        result = parser.parse_locate("(5.0, 5.0)")
        assert len(result.positions) == 0

    def test_parse_count_property(self, parser):
        """Count property returns number of positions."""
        result = parser.parse_locate("(0.18, 0.67), (0.40, 0.29)")
        assert result.count == 2

    def test_clamps_slightly_over_one(self, parser):
        """Values slightly over 1.0 should be clamped to 1.0."""
        result = parser.parse_locate("(1.05, 0.99)")
        assert len(result.positions) == 1
        assert result.positions[0] == (1.0, 0.99)

    def test_accepts_edge_values(self, parser):
        """Accept values at exactly 0 and 1."""
        result = parser.parse_locate("(0, 0), (1, 1), (0.5, 0.5)")
        assert len(result.positions) == 3
        assert (0.0, 0.0) in result.positions
        assert (1.0, 1.0) in result.positions
        assert (0.5, 0.5) in result.positions


class TestParsePattern:
    """Tests for PATTERN response parsing."""

    @pytest.fixture
    def parser(self):
        return ResponseParser()

    def test_parse_exact_match(self, parser):
        """Parse exact pattern name."""
        for pattern in ["RANDOM", "HEXAGONAL", "GRID", "EMPTY", "SINGLE"]:
            result = parser.parse_pattern(pattern)
            assert result.pattern == pattern
            assert result.success

    def test_parse_lowercase(self, parser):
        """Parse lowercase pattern names."""
        result = parser.parse_pattern("hexagonal")
        assert result.pattern == "HEXAGONAL"

    def test_parse_at_start(self, parser):
        """Parse pattern at start of response."""
        result = parser.parse_pattern("RANDOM arrangement of spots")
        assert result.pattern == "RANDOM"

    def test_parse_in_sentence(self, parser):
        """Parse pattern in a sentence."""
        result = parser.parse_pattern("The pattern appears to be HEXAGONAL")
        assert result.pattern == "HEXAGONAL"

    def test_parse_synonym_honeycomb(self, parser):
        """Parse honeycomb as hexagonal."""
        result = parser.parse_pattern("It's a honeycomb pattern")
        assert result.pattern == "HEXAGONAL"

    def test_parse_synonym_scattered(self, parser):
        """Parse scattered as random."""
        result = parser.parse_pattern("The spots are scattered randomly")
        assert result.pattern == "RANDOM"

    def test_parse_synonym_blank(self, parser):
        """Parse blank as empty."""
        result = parser.parse_pattern("The image is blank")
        assert result.pattern == "EMPTY"

    def test_parse_empty_response(self, parser):
        """Empty response should fail."""
        result = parser.parse_pattern("")
        assert result.pattern is None
        assert not result.success

    def test_parse_unknown_pattern(self, parser):
        """Unknown pattern should fail."""
        result = parser.parse_pattern("circular arrangement")
        assert result.pattern is None


class TestParseSize:
    """Tests for SIZE response parsing."""

    @pytest.fixture
    def parser(self):
        return ResponseParser()

    def test_parse_just_number(self, parser):
        """Parse response that is just a number."""
        result = parser.parse_size("3.5")
        assert result.diameter_um == 3.5
        assert result.success

    def test_parse_with_um_unit(self, parser):
        """Parse number with um units."""
        result = parser.parse_size("3.5 um")
        assert result.diameter_um == 3.5

    def test_parse_with_micrometers(self, parser):
        """Parse number with micrometers."""
        result = parser.parse_size("3.5 micrometers")
        assert result.diameter_um == 3.5

    def test_parse_labeled_diameter(self, parser):
        """Parse 'diameter: X' format."""
        result = parser.parse_size("diameter: 4.2")
        assert result.diameter_um == 4.2

    def test_parse_approximate(self, parser):
        """Parse 'approximately X' format."""
        result = parser.parse_size("approximately 5.0 um")
        assert result.diameter_um == 5.0

    def test_parse_empty_response(self, parser):
        """Empty response should fail."""
        result = parser.parse_size("")
        assert result.diameter_um is None
        assert not result.success

    def test_rejects_huge_numbers(self, parser):
        """Reject unreasonably large diameters."""
        result = parser.parse_size("150 um")
        assert result.diameter_um is None


class TestParseDefect:
    """Tests for DEFECT response parsing."""

    @pytest.fixture
    def parser(self):
        return ResponseParser()

    def test_parse_structured_yes(self, parser):
        """Parse structured response with defects."""
        response = """
        DEFECTS_FOUND: YES
        MISSING_COUNT: 3
        NOISE_COUNT: 2
        CONFIDENCE: HIGH
        """
        result = parser.parse_defect(response)
        assert result.has_defects is True
        assert result.missing_count == 3
        assert result.noise_count == 2
        assert result.confidence == "HIGH"
        assert result.success

    def test_parse_structured_no(self, parser):
        """Parse structured response without defects."""
        response = """
        DEFECTS_FOUND: NO
        CONFIDENCE: HIGH
        """
        result = parser.parse_defect(response)
        assert result.has_defects is False
        assert result.missing_count == 0
        assert result.noise_count == 0
        assert result.success

    def test_parse_natural_no_defects(self, parser):
        """Parse natural language saying no defects."""
        result = parser.parse_defect("The pattern is perfect with no defects")
        assert result.has_defects is False

    def test_parse_natural_has_defects(self, parser):
        """Parse natural language indicating defects."""
        result = parser.parse_defect("Defects detected in the pattern")
        assert result.has_defects is True

    def test_parse_empty_response(self, parser):
        """Empty response should fail."""
        result = parser.parse_defect("")
        assert result.has_defects is None
        assert not result.success


class TestParseDispatch:
    """Tests for the generic parse() method."""

    @pytest.fixture
    def parser(self):
        return ResponseParser()

    def test_dispatch_count(self, parser):
        """Dispatch to parse_count."""
        result = parser.parse("COUNT", "42")
        assert isinstance(result, ParsedCountResponse)
        assert result.count == 42

    def test_dispatch_locate(self, parser):
        """Dispatch to parse_locate."""
        result = parser.parse("LOCATE", "(100, 200)")
        assert isinstance(result, ParsedLocateResponse)

    def test_dispatch_pattern(self, parser):
        """Dispatch to parse_pattern."""
        result = parser.parse("PATTERN", "HEXAGONAL")
        assert isinstance(result, ParsedPatternResponse)

    def test_dispatch_size(self, parser):
        """Dispatch to parse_size."""
        result = parser.parse("SIZE", "3.5")
        assert isinstance(result, ParsedSizeResponse)

    def test_dispatch_defect(self, parser):
        """Dispatch to parse_defect."""
        result = parser.parse("DEFECT", "DEFECTS_FOUND: NO")
        assert isinstance(result, ParsedDefectResponse)

    def test_dispatch_case_insensitive(self, parser):
        """Task names are case-insensitive."""
        result = parser.parse("count", "42")
        assert isinstance(result, ParsedCountResponse)

    def test_dispatch_invalid_task(self, parser):
        """Invalid task raises ValueError."""
        with pytest.raises(ValueError, match="Unknown task"):
            parser.parse("INVALID", "42")
