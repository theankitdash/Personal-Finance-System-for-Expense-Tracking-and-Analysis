import sys
import json

def financial_analysis(data):
    # Parse JSON data
    expenses_data = json.loads(data)

    # Perform analysis on expenses data
    total_amount = sum(entry['total_amount'] for entry in expenses_data)
    average_amount = total_amount / len(expenses_data)

    # Example analysis: Return total amount and average amount
    return {'total_amount': total_amount, 'average_amount': average_amount}

if __name__ == "__main__":
    # Read financial data from command line argument
    data = sys.argv[1]

    # Perform financial analysis
    result = financial_analysis(data)

    # Print the result
    print(json.dumps(result))
