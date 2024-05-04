import sys
import json

def financial_analysis(data):
    # Parse JSON data
    expenses_data = json.loads(data)

    # Perform analysis on expenses data
    total_amount = sum(entry['total_amount'] for entry in expenses_data)
    average_amount = total_amount / len(expenses_data)

    # Construct analysis result message
    analysis_result = f"Total Amount: {total_amount}, Average Amount: {average_amount}"

    return analysis_result
    
if __name__ == "__main__":
    # Read financial data from command line argument
    data = sys.argv[1]

    # Perform financial analysis
    result = financial_analysis(data)

    # Print the result
    print(result)
