import sys
import json
import pandas as pd
from datetime import timedelta, datetime

def analyze_financial_data(aggregated_data, budget_data):
    # Convert lists to DataFrames
    df_aggregated = pd.DataFrame(aggregated_data)
    df_budget = pd.DataFrame(budget_data)

    # Aggregate Data Analysis
    try:
        df_aggregated['date'] = pd.to_datetime(df_aggregated['date']) + timedelta(days=1)
        df_aggregated['month'] = df_aggregated['date'].dt.strftime('%b-%Y')
        df_aggregated = df_aggregated[['date', 'month', 'category', 'amount', 'description']]

        # Summarize by month and category
        month_category_data = df_aggregated.groupby(['month', 'category'])['amount'].sum().unstack(fill_value=0) 
    except KeyError as e:
        print(f"Missing column in aggregated data: {e}", file=sys.stderr)
        month_category_data = pd.DataFrame()
        
    # Generate report lines
    report_lines = []
    report_lines.append(" ")
    
    # Include the data categorized by month    
    if not month_category_data.empty:
        current_month = datetime.now().strftime('%b-%Y')

        for month, categories in month_category_data.iterrows():
            report_lines.append(f"Month: {month}")
            total_spent_in_month = categories.sum()

            for category, amount in categories.items():
                budget_amount = df_budget.loc[df_budget['category'] == category, 'amount'].values
                budget_amount = budget_amount[0] if len(budget_amount) > 0 else None
                
                if budget_amount is not None:
                    if amount > budget_amount:
                        status = f"Over Budget by INR {amount - budget_amount:.2f}"
                    else:
                        remaining_budget = budget_amount - amount
                        percent_left = (remaining_budget / budget_amount) * 100
                        status = f"Within Budget (Spent INR {amount:.2f} of INR {budget_amount:.2f})"
                else:
                    status = f"No budget set for this category. Spent INR {amount:.2f}"
                
                report_lines.append(f"  Category: {category}, {status}")

            # If it's the current month, add extra information
            if month == current_month:
                total_budget_current_month = df_budget['amount'].sum()
                remaining_budget_current_month = total_budget_current_month - total_spent_in_month
                percent_left_current_month = (remaining_budget_current_month / total_budget_current_month) * 100
                report_lines.append(f"  Total Budget for {month}: INR {total_budget_current_month:.2f}")
                report_lines.append(f"  Total Spent: INR {total_spent_in_month:.2f}")
                report_lines.append(f"  {percent_left_current_month:.2f}% of the total budget is left for the month.")
            
            # Add the total spent in the month
            report_lines.append(f"Total Amount Spent in {month}: INR {total_spent_in_month:.2f}")
            report_lines.append("")  # Blank line for separation

        # Calculate the total amount spent
        total_amount_dt = month_category_data.sum().sum()
        report_lines.append(f"Total Amount Spent: INR {total_amount_dt:.2f}")
    else:
        report_lines.append("No financial data available for the given period.")        
    
    return "\n".join(report_lines), month_category_data
    

def main():
    try:
        input_data = json.load(sys.stdin)
        
        aggregated_data = input_data.get('aggregatedData', [])
        budget_data = input_data.get('budgets', [])

        # Analyze financial data
        report,_ = analyze_financial_data(aggregated_data, budget_data)
        print(report)  # Print the report lines
        
    except json.JSONDecodeError:
        print("Invalid JSON data provided.", file=sys.stderr)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
   

if __name__ == "__main__":
    main()
