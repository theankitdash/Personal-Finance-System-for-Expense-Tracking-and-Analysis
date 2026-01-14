import os
import asyncpg
from dotenv import load_dotenv

load_dotenv()

class Config:
    PG_USER = os.getenv("PG_USER")
    PG_PASSWORD = os.getenv("PG_PASSWORD")
    PG_DATABASE = os.getenv("PG_DATABASE")
    PG_HOST = os.getenv("PG_HOST")
    PG_PORT = os.getenv("PG_PORT")
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    JWT_SECRET = os.getenv("JWT_SECRET")

config = Config()

async def get_db_connection():
    return await asyncpg.connect(
        user=config.PG_USER,
        password=config.PG_PASSWORD,
        database=config.PG_DATABASE,
        host=config.PG_HOST,
        port=config.PG_PORT,
    )
