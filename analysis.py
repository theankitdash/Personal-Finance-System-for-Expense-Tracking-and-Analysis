import sys
import json
from collections import defaultdict
from dateutil import parser

def analyze_financial_data(data):
    month_category_data = defaultdict(lambda: defaultdict(float))
    
    # Organize data by month and category
    for item in data:
        try:
            month = parser.parse(item['date']).strftime('%Y-%m')
            month_category_data[month][item['category']] += item['amount']
        except (ValueError, KeyError, TypeError):
            print(f"Skipping invalid item: {item}")

    # Generate report lines
    report_lines = []
    for month, categories in sorted(month_category_data.items()):
        report_lines.append(f"Month: {month}")
        for category, amount in sorted(categories.items()):
            report_lines.append(f"  Category: {category}, Amount: INR {amount:.2f}")
        report_lines.append("")  # Blank line for separation
    
    total_amount = sum(amount for categories in month_category_data.values() for amount in categories.values())
    report_lines.append(f"Total Amount Spent: INR {total_amount:.2f}")
    
    return "\n".join(report_lines)

def main():
    if len(sys.argv) > 1:
        try:
            data = json.loads(sys.argv[1])
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
