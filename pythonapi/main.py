from fastapi import FastAPI
from pythonapi.analysis import analyze_financial_data
#from analysis import analyze_financial_data
from fastapi.middleware.cors import CORSMiddleware
from .config_db import config
from .router import router

app = FastAPI()

app.add_middleware(
            CORSMiddleware,
            allow_origins=config.ALLOWED_ORIGINS, 
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

app.include_router(router)

# Run the application using: uvicorn pythonapi.main:app --reload