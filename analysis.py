import sys
import json
import pandas as pd
from datetime import timedelta, datetime

def analyze_financial_data(budget_data, from_date, to_date, aggregated_data):
    # Convert lists to DataFrames
    df_aggregated = pd.DataFrame(aggregated_data)
    df_budget = pd.DataFrame(budget_data)

    # Aggregate Data Analysis
    df_aggregated['date'] = pd.to_datetime(df_aggregated['date']) + timedelta(days=1)
    df_aggregated['month'] = df_aggregated['date'].dt.strftime('%b-%Y')
    df_aggregated = df_aggregated[['date', 'month', 'category', 'amount', 'description']]

    # Filter data within the date range
    df_aggregated = df_aggregated[(df_aggregated['date'] >= from_date) & (df_aggregated['date'] <= to_date)]

    # Summarize by month and category
    try:
        month_category_data = df_aggregated.groupby(['month', 'category'])['amount'].sum().unstack(fill_value=0) 
    except KeyError as e:
        print(f"Missing column in aggregated data: {e}", file=sys.stderr)
        month_category_data = pd.DataFrame()

    # Generate report lines
    report_lines = []
    report_lines.append(f"Report from {from_date} to {to_date}")
    report_lines.append(" ")

    month_wise_spending = {}
    total_spending = 0
    total_budget_current_month = 0
    total_spent_current_month = 0
    current_month = datetime.now().strftime('%b-%Y')

    # Calculate total budget for the current month by summing the budget for all categories
    total_budget_current_month = df_budget['amount'].sum()
    
    # Include the data categorized by month    
    if not month_category_data.empty:
        for month, categories in month_category_data.iterrows():
            report_lines.append(f"Month: {month}")
            total_spent_in_month = categories.sum()
            month_wise_spending[month] = total_spent_in_month

            for category, amount in categories.items():
                budget_amount = df_budget.loc[df_budget['category'] == category, 'amount'].values
                budget_amount = budget_amount[0] if len(budget_amount) > 0 else None
                
                if budget_amount is not None and pd.notnull(budget_amount):
                    if amount > budget_amount:
                        status = f"Over Budget by INR {amount - budget_amount:.2f}"
                    else:
                        remaining_budget = budget_amount - amount
                        status = f"Within Budget (Spent INR {amount:.2f} of INR {budget_amount:.2f})"
                else:
                    status = f"No budget set for this category. Spent INR {amount:.2f}"
                    remaining_budget = None

                report_lines.append(f"  Category: {category}, {status}")

            # Add the total spent in the month
            report_lines.append(f"Total Amount Spent in {month}: INR {total_spent_in_month:.2f}")
            report_lines.append("")  # Blank line for separation

        # Calculate the total spending for the current month
        total_spent_current_month = month_category_data.loc[current_month].sum() if current_month in month_category_data.index else 0

        # Report the amount of budget left for the current month
        amount_budget_left = total_budget_current_month - total_spent_current_month
        
        # Report the percentage of budget left for the current month
        if total_budget_current_month > 0:
            report_lines.append(f"Total Budget for Current Month: INR {total_budget_current_month:.2f}")
            report_lines.append(f"Total Amount Spent in Current Month: INR {total_spent_current_month:.2f}")
            report_lines.append(f"Amount of Budget Left for Current Month ({current_month}): INR {amount_budget_left:.2f}")
        else:
            report_lines.append(f"No budget information available for the current month ({current_month}).")

        report_lines.append("") 
        
        # Calculate the total amount spent across all months within the range
        total_spending = month_category_data.sum().sum()
        report_lines.append(f"Total Amount Spent from {from_date} to {to_date}: INR {total_spending:.2f}")    
    else:
        report_lines.append("No financial data available for the given period.")

    return report_lines    
    

def main():
    try:
        input_data = json.load(sys.stdin)
        
        budget_data = input_data.get('budgets', [])
        from_date = input_data.get('fromDate', '')
        to_date = input_data.get('toDate', '')
        aggregated_data = input_data.get('aggregatedData', [])
    
        # Perform financial data analysis
        report_lines = analyze_financial_data(budget_data, from_date, to_date, aggregated_data)
        
        # Print the report
        if report_lines:
            for line in report_lines:
                print(line)
        else:
            print("No report generated.")    
        
    except json.JSONDecodeError:
        print("Invalid JSON data provided.", file=sys.stderr)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
   

if __name__ == "__main__":
    main()
