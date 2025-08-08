import sys
import json
import pandas as pd
from datetime import timedelta, datetime
from sklearn.ensemble import IsolationForest

# Function to convert and prepare data for analysis
def prepare_data(aggregated_data, budget_data):
    df_aggregated = pd.DataFrame(aggregated_data)
    df_budget = pd.DataFrame(budget_data)

    # Convert 'amount' columns to float
    df_aggregated['amount'] = df_aggregated['amount'].astype(float)
    df_budget['amount'] = df_budget['amount'].astype(float)
    
    # Process date column and create month column
    df_aggregated['date'] = pd.to_datetime(df_aggregated['date']) + timedelta(days=1)
    df_aggregated['month'] = df_aggregated['date'].dt.strftime('%b-%Y')
    
    return df_aggregated, df_budget

# Function to filter data by date range
def filter_by_date(df_aggregated, from_date, to_date):
    return df_aggregated[(df_aggregated['date'] >= from_date) & (df_aggregated['date'] <= to_date)]

# Function to group data by month and category
def group_by_month_category(df_aggregated):
    try:
        month_category_data = df_aggregated.groupby(['month', 'category'])['amount'].sum().unstack(fill_value=0)
    except KeyError as e:
        print(f"Missing column in aggregated data: {e}", file=sys.stderr)
        month_category_data = pd.DataFrame()
    
    return month_category_data

# ML-based anomaly detection using Isolation Forest
def detect_anomalies_isolation_forest(df_aggregated):
    anomalies = []

    # Prepare data: group by month & category
    df_grouped = df_aggregated.groupby(['month', 'category'])['amount'].sum().reset_index()

    # Encode categorical columns
    df_grouped['month_num'] = pd.factorize(df_grouped['month'])[0]
    df_grouped['category_num'] = pd.factorize(df_grouped['category'])[0]

    # Feature set for model
    X = df_grouped[['month_num', 'category_num', 'amount']]

    # Apply Isolation Forest
    iso = IsolationForest(n_estimators=200, contamination=0.1, max_samples='auto', random_state=42)
    df_grouped['anomaly'] = iso.fit_predict(X)

    # Filter anomalies
    anomalies_df = df_grouped[df_grouped['anomaly'] == -1]

    for _, row in anomalies_df.iterrows():
        anomalies.append({
            'month': row['month'],
            'category': row['category'],
            'amount': row['amount']
        })

    return anomalies

# Function to generate report for a single month
def generate_month_report(month, categories, df_budget, report_lines):
    total_spent_in_month = categories.sum()
    report_lines.append(f"Month: {month}")
    
    for category, amount in categories.items():
        budget_amount = df_budget.loc[df_budget['category'] == category, 'amount'].values
        budget_amount = budget_amount[0] if len(budget_amount) > 0 else None

        if budget_amount is not None and pd.notnull(budget_amount):
            if amount > budget_amount:
                over_budget = amount - budget_amount
                status = f"Over Budget by INR {over_budget:.2f} (Spent INR {amount:.2f} of INR {budget_amount:.2f}"
            else:
                status = f"Within Budget (Spent INR {amount:.2f} of INR {budget_amount:.2f})"
        else:
            status = f"No budget set for this category. Spent INR {amount:.2f}"
        
        report_lines.append(f"  Category: {category}, {status}")

    report_lines.append(f"Total Amount Spent in {month}: INR {total_spent_in_month:.2f}")
    report_lines.append("")  # Blank line for separation

# Function to generate the full financial report
def generate_report(df_budget, month_category_data, from_date, to_date, df_aggregated):
    report_lines = []
    report_lines.append(f"Report from {from_date} to {to_date}")
    report_lines.append(" ")
    
    month_wise_spending = {}
    total_spending = 0
    total_budget_current_month = df_budget['amount'].sum()
    current_month = datetime.now().strftime('%b-%Y')
    total_spent_current_month = 0

    # Generate report for each month
    if not month_category_data.empty:
        for month, categories in month_category_data.iterrows():
            generate_month_report(month, categories, df_budget, report_lines)
            month_wise_spending[month] = categories.sum()

        # Only show current month budget if the current month is in the filtered data range
        if current_month in month_category_data.index:
            total_spent_current_month = month_category_data.loc[current_month].sum()
            amount_budget_left = total_budget_current_month - total_spent_current_month
            
            report_lines.append(f"Total Budget for Current Month: INR {total_budget_current_month:.2f}")
            report_lines.append(f"Total Amount Spent in Current Month: INR {total_spent_current_month:.2f}")
            report_lines.append(f"Amount of Budget Left for Current Month ({current_month}): INR {amount_budget_left:.2f}")
            report_lines.append("")  # Blank line for separation
        
        total_spending = month_category_data.sum().sum()
        report_lines.append(f"Total Amount Spent from {from_date} to {to_date}: INR {total_spending:.2f}")
        report_lines.append("")

        # ML-based anomaly detection
        ml_anomalies = detect_anomalies_isolation_forest(df_aggregated)

        if ml_anomalies:
            report_lines.append("ML-based Anomaly Detection:")
            for item in ml_anomalies:
                report_lines.append(
                    f"  • {item['month']} - Category: {item['category']} had unusual spending of INR {item['amount']:.2f}"
                )
            report_lines.append("")    
    else:
        report_lines.append("No financial data available for the given period.")
    
    return report_lines

# Main function to orchestrate the analysis
def analyze_financial_data(budget_data, from_date, to_date, aggregated_data):
    df_aggregated, df_budget = prepare_data(aggregated_data, budget_data)
    df_aggregated = filter_by_date(df_aggregated, from_date, to_date)
    month_category_data = group_by_month_category(df_aggregated)
    
    return generate_report(df_budget, month_category_data, from_date, to_date, df_aggregated)       

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
