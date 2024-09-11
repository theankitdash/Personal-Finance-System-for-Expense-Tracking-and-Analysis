import sys
import json
import pandas as pd
from datetime import timedelta

def analyze_financial_data(aggregated_data):
    # Convert lists to DataFrames
    df_aggregated = pd.DataFrame(aggregated_data)

    # Aggregate Data Analysis
    try:
        df_aggregated['date'] = pd.to_datetime(df_aggregated['date']) + timedelta(days=1)
        df_aggregated['month'] = df_aggregated['date'].dt.strftime('%b-%Y')
        month_category_data = df_aggregated.groupby(['month', 'category'])['amount'].sum().unstack(fill_value=0) 
    except KeyError as e:
        print(f"Missing column in aggregated data: {e}", file=sys.stderr)
        month_category_data = pd.DataFrame()
        
    # Generate report lines
    report_lines = []
    report_lines.append(" ")
    
    # Include the data categorized by month    
    if not month_category_data.empty:
        for month, categories in month_category_data.iterrows():
            report_lines.append(f"Month: {month}")
            for category, amount in categories.items():
                if amount > 0:
                    report_lines.append(f"  Category: {category}, Amount: INR {amount:.2f}")
            report_lines.append("")  # Blank line for separation
    
        # Calculate the total amount spent
        total_amount_dt = month_category_data.sum().sum()
        report_lines.append(f"Total Amount Spent: INR {total_amount_dt:.2f}")
    
    return "\n".join(report_lines), month_category_data
    

def main():
    try:
        input_data = json.load(sys.stdin)
        
        aggregated_data = input_data.get('aggregatedData', [])

        # Analyze financial data
        report,_ = analyze_financial_data(aggregated_data)
        print(report)  # Print the report lines
        
    except json.JSONDecodeError:
        print("Invalid JSON data provided.", file=sys.stderr)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
   

if __name__ == "__main__":
    main()
