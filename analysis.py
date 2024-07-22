import sys
import json
from collections import defaultdict
from dateutil import parser

def analyze_financial_data(data):
    # Organize data by month and category
    month_category_data = defaultdict(lambda: defaultdict(float))
    
    for item in data:
        category = item['category']
        amount = item['amount']
        date = item['date']
        # Use dateutil.parser to handle ISO 8601 format
        month = parser.parse(date).strftime('%Y-%m')
        month_category_data[month][category] += amount
    
    # Generate the report
    report_lines = []
    for month, categories in sorted(month_category_data.items()):
        report_lines.append(f"Month: {month}")
        for category, amount in sorted(categories.items()):
            report_lines.append(f"  Category: {category}, Amount: ${amount:.2f}")
        report_lines.append("")  # Blank line for separation
    
    total_amount = sum(amount for categories in month_category_data.values() for amount in categories.values())
    report_lines.append(f"Total Amount Spent: ${total_amount:.2f}")
    
    return "\n".join(report_lines)

def main():
    # Read data from command-line argument
    if len(sys.argv) > 1:
        input_data = sys.argv[1]
        data = json.loads(input_data)
        report = analyze_financial_data(data)
        print(report)
    else:
        print("No data provided.")

if __name__ == "__main__":
    main()
