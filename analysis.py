import sys
import json
from collections import defaultdict
from dateutil import parser
from datetime import timedelta

def analyze_financial_data(aggregated_data, detailed_data):
    # Aggregate Data Analysis
    aggregated_totals = defaultdict(float)
    for item in aggregated_data:
        try:
            aggregated_totals[item['category']] += item['total_amount']
        except (KeyError, TypeError):
            print(f"Skipping invalid aggregated item: {item}")

    # Detailed Data Analysis
    month_category_data = defaultdict(lambda: defaultdict(float))
    for item in detailed_data:
        try:
            date = parser.parse(item['date']) + timedelta(days=1)
            month = date.strftime('%b-%Y')
            month_category_data[month][item['category']] += item['amount']
        except (ValueError, KeyError, TypeError):
            print(f"Skipping invalid detailed item: {item}")

    # Generate report lines
    report_lines = []

    # Include the aggregated totals in the report
    if aggregated_totals:
        report_lines.append("Aggregated Total:")
        for category, amount in sorted(aggregated_totals.items()):
            report_lines.append(f"  Category: {category}, Amount: INR {amount:.2f}")
        report_lines.append("")

        #  Calculate the total amount spent
        total_amount = sum(amount for amount in aggregated_totals.values()) 
        report_lines.append(f"Total Amount Spent: INR {total_amount:.2f}")
    
    '''
    # Include the detailed data categorized by month    
    for month, categories in sorted(month_category_data.items()):
        report_lines.append(f"Month: {month}")
        for category, amount in sorted(categories.items()):
            report_lines.append(f"  Category: {category}, Amount: INR {amount:.2f}")
        report_lines.append("")  # Blank line for separation
    
    #  Calculate the total amount spent
    total_amount_dt = sum(amount for categories in month_category_data.values() for amount in categories.values())
    report_lines.append(f"Total Amount Spent: INR {total_amount_dt:.2f}")

    '''    
    return "\n".join(report_lines), month_category_data

def main():
    if len(sys.argv) > 1:
        try:
            input_data = json.loads(sys.argv[1])
            aggregated_data = input_data.get('aggregatedData', [])
            detailed_data = input_data.get('detailedData', [])
            
            # Analyze financial data
            report, month_category_data = analyze_financial_data(aggregated_data, detailed_data)
            print(report)
            
        except json.JSONDecodeError:
            print("Invalid JSON data provided.")
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print("No data provided.")

if __name__ == "__main__":
    main()
