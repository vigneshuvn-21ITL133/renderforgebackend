from tests.conftest import make_png, signup

FILES = {"image": ("photo.png", make_png(), "image/png")}


async def _submit(client, headers, quality="low", output_format="glb"):
    return await client.post(
        "/generations",
        headers=headers,
        files=FILES,
        data={"quality": quality, "output_format": output_format},
    )


async def test_generation_runs_to_completion_and_is_downloadable(client, fake_inference):
    headers = await signup(client)

    submitted = await _submit(client, headers)
    assert submitted.status_code == 202
    generation_id = submitted.json()["id"]

    # The background task runs before the ASGI call returns, so the row is already settled.
    status = (await client.get(f"/generations/{generation_id}", headers=headers)).json()
    assert status["status"] == "completed"
    assert status["vertex_count"] == 8 and status["face_count"] == 12
    assert status["download_url"] == f"/generations/{generation_id}/download"

    download = await client.get(f"/generations/{generation_id}/download", headers=headers)
    assert download.status_code == 200
    assert download.content == b"glTF-test-mesh"
    assert download.headers["content-type"].startswith("model/gltf-binary")

    assert fake_inference.calls[0]["quality"] == "low"


async def test_failed_inference_marks_generation_failed(client, fake_inference):
    fake_inference.fail_with = "GPU out of memory"
    headers = await signup(client)

    generation_id = (await _submit(client, headers)).json()["id"]

    status = (await client.get(f"/generations/{generation_id}", headers=headers)).json()
    assert status["status"] == "failed"
    assert status["error_message"] == "GPU out of memory"
    assert status["download_url"] is None

    download = await client.get(f"/generations/{generation_id}/download", headers=headers)
    assert download.status_code == 409


async def test_users_cannot_see_each_others_generations(client):
    owner_headers = await signup(client, email="owner@example.com")
    other_headers = await signup(client, email="other@example.com")
    generation_id = (await _submit(client, owner_headers)).json()["id"]

    assert (await client.get(f"/generations/{generation_id}", headers=other_headers)).status_code == 404
    assert (await client.get(f"/generations/{generation_id}/download", headers=other_headers)).status_code == 404


async def test_list_returns_only_the_callers_generations(client):
    headers = await signup(client, email="lister@example.com")
    await _submit(client, headers)
    await _submit(client, headers)

    listed = (await client.get("/generations", headers=headers)).json()

    assert len(listed) == 2
    assert all(item["status"] == "completed" for item in listed)


async def test_non_image_upload_is_rejected(client):
    headers = await signup(client, email="bad@example.com")

    response = await client.post(
        "/generations",
        headers=headers,
        files={"image": ("notes.png", b"definitely not an image", "image/png")},
    )

    assert response.status_code == 400


async def test_unsupported_quality_is_rejected(client):
    headers = await signup(client, email="quality@example.com")

    response = await _submit(client, headers, quality="ultra")

    assert response.status_code == 422


async def test_generation_requires_authentication(client):
    response = await client.post("/generations", files=FILES)

    assert response.status_code == 401
