"""users and generations

Revision ID: 0001
Revises: 
"""
from alembic import op
import sqlalchemy as sa


revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('users',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('email', sa.String(length=255), nullable=False),
    sa.Column('hashed_password', sa.String(length=255), nullable=False),
    sa.Column('full_name', sa.String(length=120), nullable=True),
    sa.Column('is_active', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_table('generations',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('user_id', sa.Uuid(), nullable=False),
    sa.Column('status', sa.Enum('pending', 'processing', 'completed', 'failed', name='generationstatus', native_enum=False, length=16), nullable=False),
    sa.Column('quality', sa.String(length=8), nullable=False),
    sa.Column('output_format', sa.String(length=8), nullable=False),
    sa.Column('source_filename', sa.String(length=255), nullable=False),
    sa.Column('input_key', sa.String(length=512), nullable=True),
    sa.Column('output_key', sa.String(length=512), nullable=True),
    sa.Column('vertex_count', sa.Integer(), nullable=True),
    sa.Column('face_count', sa.Integer(), nullable=True),
    sa.Column('error_message', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_generations_status'), 'generations', ['status'], unique=False)
    op.create_index(op.f('ix_generations_user_id'), 'generations', ['user_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_generations_user_id'), table_name='generations')
    op.drop_index(op.f('ix_generations_status'), table_name='generations')
    op.drop_table('generations')
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_table('users')
