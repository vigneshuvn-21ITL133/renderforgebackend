import io

import httpx
import pytest
from PIL import Image

from app.config import Settings
from app.factory import create_app
from app.services.inference_client import InferenceError, MeshPayload

INFERENCE_KEY = "inference-test-key-1234567890"


class FakeInference:
    """Replaces the real inference service so the generation flow runs without a GPU."""

    def __init__(self) -> None:
        self.fail_with: str | None = None
        self.calls: list[dict] = []

    async def generate_mesh(self, *, image, filename, content_type, quality, output_format):
        self.calls.append({"filename": filename, "content_type": content_type, "quality": quality})
        if self.fail_with:
            raise InferenceError(self.fail_with)
        return MeshPayload(mesh=b"glTF-test-mesh", vertex_count=8, face_count=12)


@pytest.fixture
def fake_inference() -> FakeInference:
    return FakeInference()


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        jwt_secret_key="test-jwt-secret-key-that-is-long-enough-123",
        inference_api_key=INFERENCE_KEY,
        inference_base_url="http://inference.test",
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
        storage_dir=tmp_path / "storage",
        allowed_origins=["http://localhost:3000"],
        max_upload_mb=1,
    )


@pytest.fixture
async def app(settings, fake_inference):
    application = create_app(settings, inference=fake_inference)
    await application.state.database.create_all()
    yield application
    await application.state.database.dispose()


@pytest.fixture
async def client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        yield http_client


def make_png(size=(32, 32)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, "orange").save(buffer, format="PNG")
    return buffer.getvalue()


async def signup(client, email="maker@example.com", password="correct-horse-battery") -> dict[str, str]:
    response = await client.post("/auth/signup", json={"email": email, "password": password})
    assert response.status_code == 201, response.text
    tokens = response.json()
    return {"Authorization": f"Bearer {tokens['access_token']}"}
