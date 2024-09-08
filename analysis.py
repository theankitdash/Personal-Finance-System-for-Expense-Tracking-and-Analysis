import sys
import json
import pandas as pd
from datetime import timedelta

def analyze_financial_data(aggregated_data, detailed_data):
    # Convert lists to DataFrames
    df_aggregated = pd.DataFrame(aggregated_data)
    df_detailed = pd.DataFrame(detailed_data)

    # Convert date fields to datetime
    df_aggregated['start_date'] = pd.to_datetime(df_aggregated['start_date'])
    df_aggregated['end_date'] = pd.to_datetime(df_aggregated['end_date'])

    # Aggregate Data Analysis
    try:
        aggregated_totals = df_aggregated.groupby('category')['total_amount'].sum()

        # Get the overall date range and month names
        start_date = df_aggregated['start_date'].min()
        end_date = df_aggregated['end_date'].max()
        start_date_str = start_date.strftime('%d %B %Y')
        end_date_str = end_date.strftime('%d %B %Y')
    except KeyError as e:
        print(f"Missing column in aggregated data: {e}", file=sys.stderr)
        aggregated_totals = pd.Series(dtype=float)
        start_date_str = end_date_str = "N/A"

    # Detailed Data Analysis
    try:
        df_detailed['date'] = pd.to_datetime(df_detailed['date']) + timedelta(days=1)
        df_detailed['month'] = df_detailed['date'].dt.strftime('%b-%Y')
        month_category_data = df_detailed.groupby(['month', 'category'])['amount'].sum().unstack(fill_value=0) 
    except KeyError as e:
        print(f"Missing column in detailed data: {e}", file=sys.stderr)
        month_category_data = pd.DataFrame()
        
    # Generate report lines
    report_lines = []

    # Include the date range with full dates at the top of the report
    report_lines.append(f"Selected Dates: {start_date_str} to {end_date_str}")
    report_lines.append(" ")

    # Include the aggregated totals in the report
    if not aggregated_totals.empty:
        report_lines.append("Aggregated Total:")
        for category, amount in aggregated_totals.items():
            report_lines.append(f"  Category: {category}, Amount: INR {amount:.2f}")
        report_lines.append("")

        # Calculate the total amount spent
        total_amount = aggregated_totals.sum()
        report_lines.append(f"Total Amount Spent: INR {total_amount:.2f}")
    else:
        report_lines.append("No aggregated data available.")   
    
    '''
    # Include the detailed data categorized by month    
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
    '''
    return "\n".join(report_lines), month_category_data
    

def main():
    try:
        input_data = json.load(sys.stdin)
        
        aggregated_data = input_data.get('aggregatedData', [])
        detailed_data = input_data.get('detailedData', [])
        
        # Analyze financial data
        report,_ = analyze_financial_data(aggregated_data, detailed_data)
        print(report)  # Print the report lines
        
    except json.JSONDecodeError:
        print("Invalid JSON data provided.", file=sys.stderr)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
   

if __name__ == "__main__":
    main()
