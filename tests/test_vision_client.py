"""Tests for vision client module."""

import asyncio
import base64
import io
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from PIL import Image

from vlm_geometry_bench.config import EvaluationConfig
from vlm_geometry_bench.vision_client import VisionClient, VisionResponse


class TestVisionResponse:
    """Tests for VisionResponse dataclass."""

    def test_success_when_no_error(self):
        """Response is successful when no error."""
        response = VisionResponse(content="42 spots", model="llava:7b")
        assert response.success is True

    def test_not_success_when_error(self):
        """Response is not successful when error."""
        response = VisionResponse(content="", model="llava:7b", error="API error")
        assert response.success is False

    def test_input_tokens_from_usage(self):
        """Extract input tokens from usage dict."""
        response = VisionResponse(
            content="42",
            model="llava:7b",
            usage={"prompt_tokens": 100, "completion_tokens": 10}
        )
        assert response.input_tokens == 100

    def test_output_tokens_from_usage(self):
        """Extract output tokens from usage dict."""
        response = VisionResponse(
            content="42",
            model="llava:7b",
            usage={"prompt_tokens": 100, "completion_tokens": 10}
        )
        assert response.output_tokens == 10

    def test_tokens_zero_when_no_usage(self):
        """Tokens are zero when no usage dict."""
        response = VisionResponse(content="42", model="llava:7b")
        assert response.input_tokens == 0
        assert response.output_tokens == 0


class TestVisionClientImageEncoding:
    """Tests for image encoding functionality."""

    @pytest.fixture
    def client(self):
        config = EvaluationConfig()
        return VisionClient(config)

    def test_encode_pil_image(self, client):
        """Encode PIL Image to base64."""
        img = Image.new("RGB", (100, 100), color="red")
        b64_data, mime_type = client._encode_image(img)

        assert mime_type == "image/png"
        assert len(b64_data) > 0
        # Verify it's valid base64
        decoded = base64.b64decode(b64_data)
        assert len(decoded) > 0

    def test_encode_rgba_image(self, client):
        """RGBA images are converted to RGB."""
        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        b64_data, mime_type = client._encode_image(img)

        assert mime_type == "image/png"
        # Verify the encoded image can be decoded
        decoded = base64.b64decode(b64_data)
        result_img = Image.open(io.BytesIO(decoded))
        assert result_img.mode == "RGB"

    def test_encode_file_path(self, client, tmp_path):
        """Encode image from file path."""
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (100, 100), color="blue")
        img.save(img_path)

        b64_data, mime_type = client._encode_image(str(img_path))
        assert mime_type == "image/png"
        assert len(b64_data) > 0

    def test_encode_path_object(self, client, tmp_path):
        """Encode image from Path object."""
        img_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (100, 100), color="green")
        img.save(img_path)

        b64_data, mime_type = client._encode_image(img_path)
        assert mime_type == "image/jpeg"

    def test_encode_missing_file_raises(self, client):
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            client._encode_image("/nonexistent/image.png")

    def test_encode_unsupported_type_raises(self, client):
        """Unsupported type raises TypeError."""
        with pytest.raises(TypeError):
            client._encode_image(12345)


class TestVisionClientMimeTypes:
    """Tests for MIME type detection."""

    @pytest.fixture
    def client(self):
        config = EvaluationConfig()
        return VisionClient(config)

    def test_png_mime_type(self, client):
        """PNG files get correct MIME type."""
        assert client._get_mime_type("image.png") == "image/png"

    def test_jpg_mime_type(self, client):
        """JPG files get correct MIME type."""
        assert client._get_mime_type("image.jpg") == "image/jpeg"

    def test_jpeg_mime_type(self, client):
        """JPEG files get correct MIME type."""
        assert client._get_mime_type("image.jpeg") == "image/jpeg"

    def test_gif_mime_type(self, client):
        """GIF files get correct MIME type."""
        assert client._get_mime_type("image.gif") == "image/gif"

    def test_webp_mime_type(self, client):
        """WebP files get correct MIME type."""
        assert client._get_mime_type("image.webp") == "image/webp"

    def test_unknown_defaults_to_jpeg(self, client):
        """Unknown extension defaults to JPEG."""
        assert client._get_mime_type("image.bmp") == "image/jpeg"


class TestVisionClientHeaders:
    """Tests for request header building."""

    def test_ollama_headers(self):
        """Ollama backend headers."""
        config = EvaluationConfig(backend="ollama")
        client = VisionClient(config)
        headers = client._build_headers()

        assert headers["Content-Type"] == "application/json"
        assert "Authorization" not in headers

    def test_openrouter_headers(self):
        """OpenRouter backend headers include referer."""
        config = EvaluationConfig(backend="openrouter", api_key="test-key")
        client = VisionClient(config)
        headers = client._build_headers()

        assert headers["Content-Type"] == "application/json"
        assert headers["Authorization"] == "Bearer test-key"
        assert "HTTP-Referer" in headers
        assert "X-Title" in headers

    def test_api_key_in_headers(self):
        """API key added to Authorization header."""
        config = EvaluationConfig(api_key="my-secret-key")
        client = VisionClient(config)
        headers = client._build_headers()

        assert headers["Authorization"] == "Bearer my-secret-key"


class TestVisionClientMessages:
    """Tests for message building."""

    @pytest.fixture
    def client(self):
        config = EvaluationConfig()
        return VisionClient(config)

    def test_build_simple_message(self, client):
        """Build message with just image and prompt."""
        img = Image.new("RGB", (100, 100), color="red")
        messages = client._build_messages(img, "How many spots?")

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert len(messages[0]["content"]) == 2  # image + text

    def test_build_message_with_few_shot(self, client, tmp_path):
        """Build message with few-shot examples."""
        # Create example images
        img1 = Image.new("RGB", (100, 100), color="red")
        img2 = Image.new("RGB", (100, 100), color="blue")
        query_img = Image.new("RGB", (100, 100), color="green")

        examples = [
            {"image": img1, "question": "How many?", "answer": "5"},
            {"image": img2, "question": "How many?", "answer": "10"},
        ]

        messages = client._build_messages(query_img, "How many?", examples)

        # Should have: example1_user, example1_assistant, example2_user, example2_assistant, query
        assert len(messages) == 5
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "5"
        assert messages[4]["role"] == "user"


class TestVisionClientAsyncContext:
    """Tests for async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_creates_session(self):
        """Context manager creates aiohttp session."""
        config = EvaluationConfig()
        client = VisionClient(config)

        async with client:
            assert client._session is not None
            assert not client._session.closed

    @pytest.mark.asyncio
    async def test_context_manager_closes_session(self):
        """Context manager closes session on exit."""
        config = EvaluationConfig()
        client = VisionClient(config)

        async with client:
            session = client._session

        assert session.closed

    @pytest.mark.asyncio
    async def test_close_without_session(self):
        """Close is safe without session."""
        config = EvaluationConfig()
        client = VisionClient(config)
        await client.close()  # Should not raise


class TestVisionClientSendRequest:
    """Tests for send_request method with mocked responses."""

    @pytest.mark.asyncio
    async def test_successful_request(self):
        """Successful API request returns content."""
        config = EvaluationConfig()
        client = VisionClient(config)

        mock_response = {
            "choices": [{"message": {"content": "42 spots"}}],
            "model": "llava:7b",
            "usage": {"prompt_tokens": 100, "completion_tokens": 10},
        }

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value.status = 200
            mock_context.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value = mock_context

            async with client:
                img = Image.new("RGB", (100, 100))
                response = await client.send_request(img, "How many spots?")

        assert response.success
        assert response.content == "42 spots"
        assert response.input_tokens == 100

    @pytest.mark.asyncio
    async def test_api_error_returns_error_response(self):
        """API error returns error response after retries."""
        config = EvaluationConfig(retry_attempts=1)
        client = VisionClient(config)

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value.status = 500
            mock_context.__aenter__.return_value.text = AsyncMock(return_value="Internal Server Error")
            mock_post.return_value = mock_context

            async with client:
                img = Image.new("RGB", (100, 100))
                response = await client.send_request(img, "How many spots?")

        assert not response.success
        assert "500" in response.error

    @pytest.mark.asyncio
    async def test_timeout_returns_error_response(self):
        """Timeout returns error response."""
        config = EvaluationConfig(retry_attempts=1, timeout_seconds=1)
        client = VisionClient(config)

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.side_effect = asyncio.TimeoutError()

            async with client:
                img = Image.new("RGB", (100, 100))
                response = await client.send_request(img, "How many spots?")

        assert not response.success
        assert "timed out" in response.error.lower()

    @pytest.mark.asyncio
    async def test_keyboard_interrupt_propagates(self):
        """KeyboardInterrupt is not caught."""
        config = EvaluationConfig()
        client = VisionClient(config)

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.side_effect = KeyboardInterrupt()

            async with client:
                img = Image.new("RGB", (100, 100))
                with pytest.raises(KeyboardInterrupt):
                    await client.send_request(img, "How many spots?")
