# Using Neon Postgres

1. In the Neon Console, create a project (region close to you; free tier is enough). Open **Connect**, copy the connection string.
2. `cp .env.example .env` and paste it as `RF_DATABASE_URL`. Keep `?sslmode=require`. Pooled (`-pooler`) and direct strings both work.
3. `pip install -r requirements.txt`, then create the tables: `alembic upgrade head`
4. Start the API: `uvicorn app.main:app --port 8000`, then open `http://localhost:8000/health/db`. `{"status":"ok"}` means Neon is reachable.
5. In the Neon Console, **Tables** should now show `users`, `generations` and `alembic_version`. Every signup and generation appears there.

Notes
- The first request after Neon suspends an idle database can take 1–3 seconds. The app waits up to 30s and recycles connections every 4 minutes.
- Never commit `.env`. If the password leaks, reset it in the Neon Console.
- Generated meshes still live in `storage/` on the machine with the GPU. Only metadata goes to Neon.
