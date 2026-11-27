import asyncio
import uuid

import pytest

from app.models import GenerationStatus
from app.services.storage import LocalStorage
from tests.conftest import make_png, signup


async def test_storage_roundtrip_and_delete(tmp_path):
    storage = LocalStorage(tmp_path)
    await storage.write("meshes/a.glb", b"abc")
    assert await storage.read("meshes/a.glb") == b"abc"
    await storage.delete("meshes/a.glb")
    await storage.delete("meshes/a.glb")  # deleting twice is fine
    with pytest.raises(FileNotFoundError):
        await storage.read("meshes/a.glb")


async def test_storage_blocks_path_traversal(tmp_path):
    with pytest.raises(ValueError):
        await LocalStorage(tmp_path).write("../escape.txt", b"x")


async def test_queued_job_stays_pending_until_gpu_is_free(app, client):
    headers = await signup(client)
    user_id = uuid.UUID((await client.get("/auth/me", headers=headers)).json()["id"])
    service = app.state.generations
    async with app.state.database.sessionmaker() as session:
        generation = await service.create(
            session, user_id=user_id, source_filename="a.png", image=make_png(),
            extension="png", quality="low", output_format="glb",
        )
    gid = str(generation.id)

    await service._gpu_slots.acquire()  # pretend the single GPU is busy
    task = asyncio.create_task(service.process(generation.id))
    try:
        await asyncio.sleep(0.2)
        status = (await client.get(f"/generations/{gid}", headers=headers)).json()["status"]
        assert status == GenerationStatus.PENDING.value
    finally:
        service._gpu_slots.release()
    await asyncio.wait_for(task, 10)
    assert (await client.get(f"/generations/{gid}", headers=headers)).json()["status"] == "completed"


async def test_download_is_cacheable(client):
    headers = await signup(client)
    created = await client.post("/generations", headers=headers, files={"image": ("a.png", make_png(), "image/png")})
    gid = created.json()["id"]
    for _ in range(20):
        if (await client.get(f"/generations/{gid}", headers=headers)).json()["status"] == "completed":
            break
        await asyncio.sleep(0.1)
    response = await client.get(f"/generations/{gid}/download", headers=headers)
    assert response.status_code == 200
    assert response.headers["cache-control"] == "private, max-age=86400"


async def test_password_hashing_does_not_block_the_event_loop(client):
    """While a signup hashes its password, other requests must keep being served."""
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    task = asyncio.create_task(ticker())
    await signup(client, email="slow@example.com")
    task.cancel()
    assert ticks >= 3  # a blocked loop would manage almost none
