from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Dict, Any
from analysis import analyze_financial_data
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    budgets: List[Dict[str, Any]]
    fromDate: str
    toDate: str
    aggregatedData: List[Dict[str, Any]]

@app.post("/analyze")
async def analyze(data: AnalyzeRequest):
    try:
        report_lines = analyze_financial_data(
            data.budgets,
            data.fromDate,
            data.toDate,
            data.aggregatedData
        )
        return {"report": "\n".join(report_lines)}
    except Exception as e:
        return {"error": str(e)}

# Run the application using: uvicorn main:app --reload