"""Response parser for VLM outputs.

Handles parsing of various VLM response formats into structured data
for each evaluation task.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union


@dataclass
class ParsedCountResponse:
    """Parsed response for COUNT task."""

    count: Optional[int]
    raw_text: str
    parse_error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.count is not None


@dataclass
class ParsedLocateResponse:
    """Parsed response for LOCATE task."""

    positions: List[Tuple[float, float]]
    raw_text: str
    parse_error: Optional[str] = None

    @property
    def success(self) -> bool:
        return len(self.positions) > 0 or self.parse_error is None

    @property
    def count(self) -> int:
        return len(self.positions)


@dataclass
class ParsedPatternResponse:
    """Parsed response for PATTERN task."""

    pattern: Optional[str]  # RANDOM, HEXAGONAL, GRID, EMPTY, SINGLE
    raw_text: str
    parse_error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.pattern is not None


@dataclass
class ParsedSizeResponse:
    """Parsed response for SIZE task."""

    diameter_um: Optional[float]
    raw_text: str
    parse_error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.diameter_um is not None


@dataclass
class ParsedDefectResponse:
    """Parsed response for DEFECT task."""

    has_defects: Optional[bool]
    missing_count: Optional[int]
    noise_count: Optional[int]
    confidence: Optional[str]  # HIGH, MEDIUM, LOW
    raw_text: str
    parse_error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.has_defects is not None


# Type alias for any parsed response
ParsedResponse = Union[
    ParsedCountResponse,
    ParsedLocateResponse,
    ParsedPatternResponse,
    ParsedSizeResponse,
    ParsedDefectResponse,
]


class ResponseParser:
    """Parser for VLM responses across different tasks."""

    # Valid pattern types
    VALID_PATTERNS = {"RANDOM", "HEXAGONAL", "GRID", "EMPTY", "SINGLE"}

    # Valid confidence levels
    VALID_CONFIDENCE = {"HIGH", "MEDIUM", "LOW"}

    def parse_count(self, response: str) -> ParsedCountResponse:
        """Parse a COUNT task response to extract the spot count.

        Handles various formats:
        - Just a number: "42"
        - With units: "42 spots"
        - Natural language: "I can see 42 spots"
        - After colon: "Answer: 42"

        Args:
            response: Raw VLM response text

        Returns:
            ParsedCountResponse with extracted count
        """
        text = response.strip()

        # Handle empty response
        if not text:
            return ParsedCountResponse(
                count=None,
                raw_text=response,
                parse_error="Empty response"
            )

        # Try to find a number in the response
        # First, check if the entire response is just a number
        try:
            count = int(text)
            return ParsedCountResponse(count=count, raw_text=response)
        except ValueError:
            pass

        # Patterns to match, in order of specificity
        patterns = [
            r"^(\d+)$",  # Just a number
            r"^(\d+)\s*(?:spots?|circles?|dots?|points?)",  # Number followed by unit
            r"(?:there\s+(?:are|is)|i\s+(?:can\s+)?(?:see|count|find|detect))\s+(\d+)",  # Natural language
            r"(?:total|count|number|answer)[:\s]+(\d+)",  # Labeled answer
            r"(\d+)\s+(?:spots?|circles?|dots?|points?|visible)",  # Number + unit anywhere
            r":\s*(\d+)\s*$",  # After colon at end
            r"(\d+)",  # Last resort: any number
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    count = int(match.group(1))
                    # Sanity check: reject unreasonably large numbers
                    if count > 10000:
                        continue
                    return ParsedCountResponse(count=count, raw_text=response)
                except ValueError:
                    continue

        # Check for zero/none indicators
        zero_patterns = [
            r"no\s+(?:spots?|circles?|dots?|visible)",
            r"(?:zero|0)\s+(?:spots?|circles?|dots?)",
            r"(?:empty|blank|nothing)",
            r"cannot\s+(?:see|find|detect)",
            r"don'?t\s+see\s+any",
        ]
        for pattern in zero_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return ParsedCountResponse(count=0, raw_text=response)

        return ParsedCountResponse(
            count=None,
            raw_text=response,
            parse_error="Could not extract count from response"
        )

    def parse_locate(self, response: str) -> ParsedLocateResponse:
        """Parse a LOCATE task response to extract normalized coordinates.

        Handles various formats:
        - One per line: "0.18, 0.67\\n0.40, 0.29"
        - Parentheses: "(0.18, 0.67), (0.40, 0.29)"
        - JSON-like: "[[0.18, 0.67], [0.40, 0.29]]"
        - Natural language: "at position 0.18, 0.67"

        Args:
            response: Raw VLM response text

        Returns:
            ParsedLocateResponse with list of (x, y) tuples in normalized [0,1] coordinates
        """
        text = response.strip()
        positions = []

        if not text:
            return ParsedLocateResponse(
                positions=[],
                raw_text=response,
                parse_error="Empty response"
            )

        # Pattern for coordinate pairs - handles various formats
        # Matches: "0.18, 0.67" or "(0.18, 0.67)" or "[0.18, 0.67]" or "0.18,0.67"
        # Also matches integers like "0, 1" for edge cases
        coord_pattern = r"[\[\(]?\s*(\d+(?:\.\d+)?)\s*[,\s]\s*(\d+(?:\.\d+)?)\s*[\]\)]?"

        # Find all coordinate pairs
        matches = re.findall(coord_pattern, text)

        for match in matches:
            try:
                x = float(match[0])
                y = float(match[1])
                # Sanity check: normalized coordinates should be in [0, 1]
                # Allow slight overflow for values like 1.0 or models that return 1.01
                if 0 <= x <= 1.1 and 0 <= y <= 1.1:
                    # Clamp to [0, 1]
                    x = min(max(x, 0.0), 1.0)
                    y = min(max(y, 0.0), 1.0)
                    positions.append((x, y))
            except (ValueError, IndexError):
                continue

        if not positions:
            # Check if response indicates no spots
            if re.search(r"no\s+spots?|empty|none|zero", text, re.IGNORECASE):
                return ParsedLocateResponse(positions=[], raw_text=response)

            return ParsedLocateResponse(
                positions=[],
                raw_text=response,
                parse_error="Could not extract coordinates from response"
            )

        return ParsedLocateResponse(positions=positions, raw_text=response)

    def parse_pattern(self, response: str) -> ParsedPatternResponse:
        """Parse a PATTERN task response to extract pattern classification.

        Args:
            response: Raw VLM response text

        Returns:
            ParsedPatternResponse with extracted pattern type
        """
        text = response.strip().upper()

        if not text:
            return ParsedPatternResponse(
                pattern=None,
                raw_text=response,
                parse_error="Empty response"
            )

        # Check for exact match first (most reliable)
        for pattern in self.VALID_PATTERNS:
            if text == pattern:
                return ParsedPatternResponse(pattern=pattern, raw_text=response)

        # Check if pattern word appears at start of response (likely the answer)
        for pattern in self.VALID_PATTERNS:
            if text.startswith(pattern):
                return ParsedPatternResponse(pattern=pattern, raw_text=response)

        # Check if pattern word appears anywhere in response
        for pattern in self.VALID_PATTERNS:
            if pattern in text:
                return ParsedPatternResponse(pattern=pattern, raw_text=response)

        # Check for synonyms/variations
        synonym_map = {
            "HEXAGONAL": ["HONEYCOMB", "HEX ", "HEXAGON"],
            "RANDOM": ["SCATTERED", "IRREGULAR", "NO PATTERN", "RANDOMLY", "UNORDERED"],
            "GRID": ["RECTANGULAR", "SQUARE GRID", "REGULAR GRID"],
            "EMPTY": ["NO SPOTS", "BLANK", "NOTHING", "NONE VISIBLE"],
            "SINGLE": ["ONE SPOT", "ONLY ONE", "SINGLE SPOT", "1 SPOT"],
        }

        for pattern, synonyms in synonym_map.items():
            for synonym in synonyms:
                if synonym in text:
                    return ParsedPatternResponse(pattern=pattern, raw_text=response)

        return ParsedPatternResponse(
            pattern=None,
            raw_text=response,
            parse_error="Could not extract pattern type from response"
        )

    def parse_size(self, response: str) -> ParsedSizeResponse:
        """Parse a SIZE task response to extract diameter estimate.

        Handles various formats:
        - Just a number: "3.5"
        - With units: "3.5 micrometers"
        - Natural language: "approximately 3.5 um"

        Args:
            response: Raw VLM response text

        Returns:
            ParsedSizeResponse with extracted diameter
        """
        text = response.strip()

        if not text:
            return ParsedSizeResponse(
                diameter_um=None,
                raw_text=response,
                parse_error="Empty response"
            )

        # Try to parse as just a number first
        try:
            diameter = float(text)
            if 0 < diameter < 100:  # Sanity check
                return ParsedSizeResponse(diameter_um=diameter, raw_text=response)
        except ValueError:
            pass

        # Patterns to match floating point numbers
        patterns = [
            r"^(\d+(?:\.\d+)?)\s*$",  # Just a number
            r"(\d+(?:\.\d+)?)\s*(?:um|µm|micrometers?|microns?)",  # With units
            r"(?:diameter|size|width)[:\s]+(\d+(?:\.\d+)?)",  # Labeled
            r"(?:approximately|about|around|roughly)\s+(\d+(?:\.\d+)?)",  # Approximate
            r"(\d+(?:\.\d+)?)\s*(?:um|µm|micrometers?|microns?|\s|$)",  # Number with optional units
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    diameter = float(match.group(1))
                    # Sanity check: diameter should be reasonable for spots
                    if 0 < diameter < 100:
                        return ParsedSizeResponse(diameter_um=diameter, raw_text=response)
                except ValueError:
                    continue

        return ParsedSizeResponse(
            diameter_um=None,
            raw_text=response,
            parse_error="Could not extract diameter from response"
        )

    def parse_defect(self, response: str) -> ParsedDefectResponse:
        """Parse a DEFECT task response.

        Expected format:
        DEFECTS_FOUND: YES or NO
        MISSING_COUNT: [number]
        NOISE_COUNT: [number]
        CONFIDENCE: HIGH, MEDIUM, or LOW

        Args:
            response: Raw VLM response text

        Returns:
            ParsedDefectResponse with extracted defect information
        """
        text = response.strip()

        if not text:
            return ParsedDefectResponse(
                has_defects=None,
                missing_count=None,
                noise_count=None,
                confidence=None,
                raw_text=response,
                parse_error="Empty response"
            )

        has_defects = None
        missing_count = None
        noise_count = None
        confidence = None

        text_upper = text.upper()

        # Parse DEFECTS_FOUND
        defects_match = re.search(r"DEFECTS?[_\s]*(?:FOUND)?[:\s]*(YES|NO|TRUE|FALSE)", text_upper)
        if defects_match:
            value = defects_match.group(1)
            has_defects = value in ("YES", "TRUE")
        else:
            # Fallback: look for indicators
            if re.search(r"NO\s+DEFECTS?|PERFECT|NO\s+MISSING|NO\s+NOISE", text_upper):
                has_defects = False
            elif re.search(r"DEFECTS?\s+(?:FOUND|DETECTED|PRESENT)|MISSING|NOISE", text_upper):
                has_defects = True

        # Parse MISSING_COUNT
        missing_match = re.search(r"MISSING[_\s]*(?:COUNT)?[:\s]*(\d+)", text_upper)
        if missing_match:
            missing_count = int(missing_match.group(1))

        # Parse NOISE_COUNT
        noise_match = re.search(r"NOISE[_\s]*(?:COUNT)?[:\s]*(\d+)", text_upper)
        if noise_match:
            noise_count = int(noise_match.group(1))

        # Parse CONFIDENCE
        conf_match = re.search(r"CONFIDENCE[:\s]*(HIGH|MEDIUM|LOW)", text_upper)
        if conf_match:
            confidence = conf_match.group(1)

        # If we got defects=NO but no counts, assume 0
        if has_defects is False:
            if missing_count is None:
                missing_count = 0
            if noise_count is None:
                noise_count = 0

        # Determine if parse was successful enough
        parse_error = None
        if has_defects is None:
            parse_error = "Could not determine if defects are present"

        return ParsedDefectResponse(
            has_defects=has_defects,
            missing_count=missing_count,
            noise_count=noise_count,
            confidence=confidence,
            raw_text=response,
            parse_error=parse_error
        )

    def parse(self, task: str, response: str) -> ParsedResponse:
        """Parse a response based on task type.

        Args:
            task: Task identifier (COUNT, LOCATE, PATTERN, SIZE, DEFECT)
            response: Raw VLM response text

        Returns:
            Appropriate parsed response dataclass

        Raises:
            ValueError: If task is unknown
        """
        task_upper = task.upper()
        parsers = {
            "COUNT": self.parse_count,
            "LOCATE": self.parse_locate,
            "PATTERN": self.parse_pattern,
            "SIZE": self.parse_size,
            "DEFECT": self.parse_defect,
        }

        if task_upper not in parsers:
            raise ValueError(f"Unknown task: {task}. Valid tasks: {list(parsers.keys())}")

        return parsers[task_upper](response)
