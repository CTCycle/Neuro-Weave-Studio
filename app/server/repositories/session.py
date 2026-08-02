from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from configurations.settings import get_settings
from repositories.base import Base


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    settings = get_settings()
    if settings.database_url.startswith("sqlite:///"):
        path = Path(settings.database_url.removeprefix("sqlite:///"))
        path.parent.mkdir(parents=True, exist_ok=True)
    connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
    return create_engine(settings.database_url, connect_args=connect_args)


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def initialize_database() -> None:
    Base.metadata.create_all(bind=get_engine())


def session_scope() -> Iterator[Session]:
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()
