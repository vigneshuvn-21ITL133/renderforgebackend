from tests.conftest import signup

PASSWORD = "correct-horse-battery"


async def test_signup_returns_token_pair(client):
    response = await client.post("/auth/signup", json={"email": "New@Example.com", "password": PASSWORD})

    assert response.status_code == 201
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"] and body["refresh_token"]


async def test_duplicate_email_is_rejected_case_insensitively(client):
    await signup(client, email="dup@example.com")

    response = await client.post("/auth/signup", json={"email": "DUP@example.com", "password": PASSWORD})

    assert response.status_code == 409


async def test_short_password_is_rejected(client):
    response = await client.post("/auth/signup", json={"email": "a@example.com", "password": "short"})

    assert response.status_code == 422


async def test_login_succeeds_with_correct_password(client):
    await signup(client, email="login@example.com", password=PASSWORD)

    response = await client.post(
        "/auth/login", data={"username": "login@example.com", "password": PASSWORD}
    )

    assert response.status_code == 200
    assert "access_token" in response.json()


async def test_login_fails_with_wrong_password(client):
    await signup(client, email="wrong@example.com", password=PASSWORD)

    response = await client.post("/auth/login", data={"username": "wrong@example.com", "password": "nope-nope-nope"})

    assert response.status_code == 401


async def test_login_fails_for_unknown_email_with_same_message(client):
    response = await client.post("/auth/login", data={"username": "ghost@example.com", "password": PASSWORD})

    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect email or password"


async def test_me_returns_current_user(client):
    headers = await signup(client, email="me@example.com")

    response = await client.get("/auth/me", headers=headers)

    assert response.status_code == 200
    assert response.json()["email"] == "me@example.com"
    assert "hashed_password" not in response.json()


async def test_me_rejects_missing_token(client):
    response = await client.get("/auth/me")

    assert response.status_code == 401


async def test_refresh_issues_new_tokens(client):
    signup_response = await client.post("/auth/signup", json={"email": "r@example.com", "password": PASSWORD})
    refresh_token = signup_response.json()["refresh_token"]

    response = await client.post("/auth/refresh", json={"refresh_token": refresh_token})

    assert response.status_code == 200
    assert response.json()["access_token"]


async def test_refresh_rejects_access_token(client):
    signup_response = await client.post("/auth/signup", json={"email": "t@example.com", "password": PASSWORD})
    access_token = signup_response.json()["access_token"]

    response = await client.post("/auth/refresh", json={"refresh_token": access_token})

    assert response.status_code == 401
