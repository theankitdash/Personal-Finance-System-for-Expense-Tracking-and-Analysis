from fastapi import FastAPI, Cookie, HTTPException
from pydantic import BaseModel
from pythonapi.analysis import analyze_financial_data
#from analysis import analyze_financial_data
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import os
import jwt
import asyncpg
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
            CORSMiddleware,
            allow_origins=origins, 
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

class AnalyzeRequest(BaseModel):
        fromDate: str
        toDate: str

@app.post("/analyze")
async def analyze(data: AnalyzeRequest, auth_token: str = Cookie(None)):

    if not auth_token:
        raise HTTPException(status_code=401, detail="Missing auth token")

    try:
        payload = jwt.decode(auth_token, os.getenv("JWT_SECRET"), algorithms=["HS256"])
        phone = int(payload["phone"])  
    except:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    from_date = datetime.strptime(data.fromDate, "%Y-%m-%d").date()
    to_date = datetime.strptime(data.toDate, "%Y-%m-%d").date()
    
    async def connect_db():
        conn = await asyncpg.connect(
            user=os.getenv("PG_USER"),
            password=os.getenv("PG_PASSWORD"),
            database=os.getenv("PG_DATABASE"),
            host=os.getenv("PG_HOST"),
            port=os.getenv("PG_PORT")
        )

        # Query expenses for the given date range
        rows = await conn.fetch(
            "SELECT category, amount, description, date FROM expenses WHERE phone = $1 AND date >= $2 AND date <= $3",
            phone,
            from_date,
            to_date
        )

        # Convert asyncpg Record objects to list of dicts
        expenses = [dict(row) for row in rows]
        print(expenses)

        # Fetch budget for the user
        budget_rows = await conn.fetch(
            "SELECT category, amount FROM budget WHERE phone = $1",
            phone
        )
        budgets = [dict(row) for row in budget_rows]
        print(budgets)

        await conn.close()

        # Call your analysis function
        report_lines = analyze_financial_data(
            budgets,
            data.fromDate,
            data.toDate,
            expenses
        )

        return {"report": "\n".join(report_lines)}  
     
    return await connect_db()

# Run the application using: uvicorn pythonapi.main:app --reload