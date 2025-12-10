from pydantic import BaseModel

class AnalyzeRequest(BaseModel):
    fromDate: str
    toDate: str