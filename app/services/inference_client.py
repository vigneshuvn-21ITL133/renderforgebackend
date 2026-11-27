"""HTTP client for the inference service (renderforge-inference)."""

import logging
from dataclasses import dataclass

import httpx

from app.schemas import OutputFormat, Quality

logger = logging.getLogger(__name__)


class InferenceError(Exception):
    """The inference service could not produce a mesh. The message is safe to show users."""


@dataclass(frozen=True)
class MeshPayload:
    mesh: bytes
    vertex_count: int
    face_count: int


class InferenceClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout_seconds: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {"X-API-Key": api_key}
        self._timeout = timeout_seconds
        self._transport = transport  # injected in tests

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._base_url,
            headers=self._headers,
            timeout=httpx.Timeout(self._timeout, connect=10.0),
            transport=self._transport,
        )

    async def generate_mesh(
        self,
        *,
        image: bytes,
        filename: str,
        content_type: str,
        quality: Quality,
        output_format: OutputFormat,
    ) -> MeshPayload:
        """Run the full inference round trip: submit, download the mesh, then free scratch space."""
        try:
            async with self._client() as client:
                submitted = await client.post(
                    "/infer",
                    files={"image": (filename, image, content_type)},
                    data={"quality": quality, "output_format": output_format},
                )
                _raise_for_status(submitted, "submit image")
                job = submitted.json()

                downloaded = await client.get(f"/download/{job['job_id']}")
                _raise_for_status(downloaded, "download mesh")

                try:
                    await client.delete(f"/jobs/{job['job_id']}")
                except httpx.HTTPError:
                    # Scratch cleanup is best effort. The mesh is already in hand.
                    logger.warning("Could not delete inference job %s", job["job_id"])
        except httpx.ConnectError as exc:
            raise InferenceError(
                f"The 3D engine is not reachable at {self._base_url}. Start the GPU service and try again."
            ) from exc
        except httpx.TimeoutException as exc:
            raise InferenceError("The 3D engine took too long to respond. Try Fast quality or a smaller image.") from exc
        except httpx.HTTPError as exc:
            raise InferenceError(f"Lost the connection to the 3D engine ({type(exc).__name__}).") from exc
        except (KeyError, ValueError) as exc:
            raise InferenceError("The 3D engine returned an unexpected response.") from exc

        return MeshPayload(
            mesh=downloaded.content,
            vertex_count=job["vertex_count"],
            face_count=job["face_count"],
        )


def _raise_for_status(response: httpx.Response, action: str) -> None:
    if response.is_success:
        return
    detail = ""
    try:
        detail = str(response.json().get("detail", ""))
    except ValueError:
        pass
    raise InferenceError(f"Inference failed to {action} ({response.status_code}){': ' + detail if detail else ''}")
