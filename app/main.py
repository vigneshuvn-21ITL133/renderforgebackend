"""Entry point: ``uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload``."""

from app.factory import create_app

app = create_app()
