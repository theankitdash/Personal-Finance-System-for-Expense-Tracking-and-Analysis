import sys
import json
from collections import defaultdict
from dateutil import parser

def analyze_financial_data(data):
    # Organize data by month and category using a defaultdict of defaultdicts
    month_category_data = defaultdict(lambda: defaultdict(float))
    
    for item in data:
        category = item['category']
        amount = item['amount']
        date = item['date']
        # Parse date and extract year-month format
        month = parser.parse(date).strftime('%Y-%m')
        # Accumulate amount for each category per month
        month_category_data[month][category] += amount
    
    # Generate the report as a list of lines
    report_lines = []
    for month, categories in sorted(month_category_data.items()):
        report_lines.append(f"Month: {month}")
        for category, amount in sorted(categories.items()):
            report_lines.append(f"  Category: {category}, Amount: INR {amount:.2f}")
        report_lines.append("")  # Blank line for separation
    
    # Calculate total amount spent
    total_amount = sum(amount for categories in month_category_data.values() for amount in categories.values())
    report_lines.append(f"Total Amount Spent: INR {total_amount:.2f}")
    
    # Join all lines with newline characters
    return "\n".join(report_lines)

def main():
    # Read data from command-line argument
    if len(sys.argv) > 1:
        try:
            # Parse JSON data
            data = json.loads(sys.argv[1])
            # Analyze financial data and print the report
            report = analyze_financial_data(data)
            print(report)
        except json.JSONDecodeError:
            print("Invalid JSON data provided.")
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print("No data provided.")

if __name__ == "__main__":
    main()
