# Database migrations (Alembic)

For local/dev the application calls `init_db()` (SQLAlchemy `create_all`) on
startup, which is sufficient for SQLite and a fresh Postgres. For production,
manage the schema with Alembic:

```bash
# generate an initial migration from the ORM models
uv run alembic revision --autogenerate -m "init"

# apply migrations
uv run alembic upgrade head
```

The migration environment (`migrations/env.py`) reads `DATABASE_URL` from settings
and targets `packages.common.db:Base.metadata`.
