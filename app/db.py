# app/db.py

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from .config import settings

# Create an asynchronous engine
engine = create_async_engine(settings.DATABASE_URL, echo=False)

# Create a session maker for creating new async sessions
AsyncSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    class_=AsyncSession
)

# Base class for declarative models
Base = declarative_base()

# Dependency to get a DB session in path operations
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session