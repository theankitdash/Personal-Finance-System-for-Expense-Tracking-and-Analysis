from fastapi import APIRouter, Cookie, HTTPException
import jwt
from datetime import datetime
from .config_db import config, get_db_connection
from .models import AnalyzeRequest
from .analysis import analyze_financial_data

router = APIRouter()

@router.post("/analyze")
async def analyze(data: AnalyzeRequest, auth_token: str = Cookie(None)):

    if not auth_token:
        raise HTTPException(status_code=401, detail="Missing auth token")

    try:
        payload = jwt.decode(auth_token, config.JWT_SECRET, algorithms=["HS256"])
        phone = int(payload["phone"])
    except:
        raise HTTPException(status_code=401, detail="Invalid token")

    from_date = datetime.strptime(data.fromDate, "%Y-%m-%d").date()
    to_date = datetime.strptime(data.toDate, "%Y-%m-%d").date()

    conn = await get_db_connection()

    rows = await conn.fetch(
        "SELECT category, amount, description, date FROM expenses WHERE phone = $1",
        phone
    )
    all_expenses = [dict(r) for r in rows]

    rows = await conn.fetch(
        "SELECT category, amount, description, date FROM expenses WHERE phone = $1 AND date >= $2 AND date <= $3",
        phone, from_date, to_date
    )
    range_expenses = [dict(r) for r in rows]

    budget_rows = await conn.fetch(
        "SELECT category, amount FROM budget WHERE phone = $1",
        phone
    )
    budgets = [dict(r) for r in budget_rows]

    await conn.close()

    report = analyze_financial_data(
        budgets,
        data.fromDate,
        data.toDate,
        range_expenses,
        all_expenses
    )

    print(report)
    return report

