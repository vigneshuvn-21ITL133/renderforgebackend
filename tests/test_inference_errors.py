import httpx
import pytest

from app.services.inference_client import InferenceClient, InferenceError


def _client(handler):
    return InferenceClient("http://engine.test", "k" * 20, 5, transport=httpx.MockTransport(handler))


async def _call(client):
    return await client.generate_mesh(image=b"x", filename="a.png", content_type="image/png", quality="low", output_format="glb")


async def test_unreachable_engine_has_clear_message():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(InferenceError, match="not reachable"):
        await _call(_client(handler))


async def test_timeout_has_clear_message():
    def handler(request):
        raise httpx.ReadTimeout("slow")

    with pytest.raises(InferenceError, match="too long"):
        await _call(_client(handler))
