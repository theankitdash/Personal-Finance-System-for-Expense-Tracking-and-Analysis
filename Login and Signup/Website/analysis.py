# analysis.py

import sys
import json

def financial_analysis(data):
    # Perform financial analysis here
    # For example, calculate the total amount of expenses
    total_expenses = sum(entry['amount'] for entry in data)
    return total_expenses

if __name__ == "__main__":
    # Read financial data from command line argument
    data = json.loads(sys.argv[1])

    # Perform financial analysis
    result = financial_analysis(data)

    # Print the result
    print(result)
