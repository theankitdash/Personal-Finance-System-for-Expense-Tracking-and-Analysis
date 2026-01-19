from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config.config_db import config
from .routes.router import router

app = FastAPI()

app.add_middleware(
            CORSMiddleware,
            allow_origins=config.ALLOWED_ORIGINS, 
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

app.include_router(router)

# Health check endpoint for Docker
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "python-api"}


# Run the application using: uvicorn pythonapi.main:app --reload 