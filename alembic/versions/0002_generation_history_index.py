"""index for the per-user history list

Revision ID: 0002
Revises: 0001
"""
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index("ix_generations_user_created", "generations", ["user_id", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_generations_user_created", table_name="generations")
