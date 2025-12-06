import sys
import json
import pandas as pd
from datetime import timedelta, datetime
from pythonapi.ml_module import FinanceML

# Function to convert and prepare data for analysis
def prepare_data(aggregated_data, budget_data):
    df_aggregated = pd.DataFrame(aggregated_data)
    df_budget = pd.DataFrame(budget_data)
    
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

# ML-based financial analysis using FinanceML module
def analyze_with_ml(df_aggregated, df_budget):
    """
    Use FinanceML module for comprehensive financial analysis including:
    - Anomaly detection (LOF, One-Class SVM, Autoencoder)
    - Drift detection
    - Spending predictions
    - Category clustering
    """
    ml_insights = {
        'anomalies': [],
        'drift_report': None,
        'predictions': None,
        'category_clusters': None
    }
    
    try:
        # Convert Decimal types to float (from MongoDB)
        df_aggregated_clean = df_aggregated.copy()
        for col in df_aggregated_clean.columns:
            if df_aggregated_clean[col].dtype == object:
                try:
                    df_aggregated_clean[col] = df_aggregated_clean[col].astype(float)
                except (ValueError, TypeError):
                    pass
        
        df_budget_clean = df_budget.copy()
        for col in df_budget_clean.columns:
            if df_budget_clean[col].dtype == object:
                try:
                    df_budget_clean[col] = df_budget_clean[col].astype(float)
                except (ValueError, TypeError):
                    pass
        
        # Initialize FinanceML
        fm = FinanceML(model_dir='ml_models')
        
        # Build features from the data
        features_df = fm.build_features(df_aggregated_clean, df_budget_clean)
        
        if features_df.empty:
            return ml_insights
        
        # 1. Fit unsupervised anomaly detection models
        fm.fit_unsupervised(features_df)
        
        # Detect anomalies
        anomalies_df = fm.detect_anomalies(features_df, return_scores=True)
        anomalies_df = anomalies_df[anomalies_df['is_anomaly']]
        
        for _, row in anomalies_df.iterrows():
            ml_insights['anomalies'].append({
                'month': row['month_str'],
                'category': row['category'],
                'amount': float(row['amount']),
                'anomaly_votes': int(row['anomaly_votes']),
                'lof_score': float(row['lof_score']) if pd.notna(row['lof_score']) else None,
                'ocsvm_score': float(row['ocsvm_score']) if pd.notna(row['ocsvm_score']) else None,
                'ae_mse': float(row['ae_mse']) if pd.notna(row['ae_mse']) else None
            })
        
        # 2. Compute drift detection
        ml_insights['drift_report'] = fm.compute_drift(df_aggregated_clean)
        
        # 3. Train regressors and predict next month
        fm.train_regressors(df_aggregated_clean, df_budget_clean)
        predictions = fm.predict_next_month(df_aggregated_clean, df_budget_clean)
        
        ml_insights['predictions'] = {
            cat: {
                'pred': float(data['pred']),
                'model': data['model'],
                'score': float(data['score']),
                'conf_int': (float(data['conf_int'][0]), float(data['conf_int'][1]))
            }
            for cat, data in predictions.items()
        }
        
        # 4. Semantic category clustering (optional)
        try:
            categories = df_aggregated_clean['category'].unique().tolist()
            if len(categories) > 1:
                fm.fit_category_embeddings(categories)
                n_clusters = max(2, min(len(categories) // 3, 8))
                clusters = fm.cluster_categories(n_clusters=n_clusters)
                ml_insights['category_clusters'] = clusters
        except Exception:
            pass  # Category clustering is optional
    
    except Exception as e:
        print(f"Warning: FinanceML analysis failed: {e}. Continuing with basic analysis.", file=sys.stderr)
    
    return ml_insights

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

        # ML-based comprehensive financial analysis
        ml_insights = analyze_with_ml(df_aggregated, df_budget)

        # Anomaly Detection Section
        if ml_insights['anomalies']:
            report_lines.append("ML-based Anomaly Detection:")
            for item in ml_insights['anomalies']:
                report_lines.append(
                    f"  • {item['month']} - Category: {item['category']} had unusual spending of INR {item['amount']:.2f} (Votes: {item['anomaly_votes']})"
                )
            report_lines.append("")
        
        # Drift Detection Section
        if ml_insights['drift_report'] and 'message' not in ml_insights['drift_report']:
            report_lines.append("Spending Pattern Drift Detection:")
            drift = ml_insights['drift_report']
            report_lines.append(f"  Jensen-Shannon Distance: {drift['jensen_shannon']:.4f}")
            if drift['top_psi_contributors']:
                report_lines.append("  Top Drifting Categories:")
                for cat, psi in drift['top_psi_contributors'][:5]:
                    report_lines.append(f"    - {cat}: PSI = {psi:.4f}")
            report_lines.append("")
        
        # Spending Predictions Section
        if ml_insights['predictions']:
            report_lines.append("Next Month Spending Predictions:")
            for cat, pred_data in ml_insights['predictions'].items():
                low, high = pred_data['conf_int']
                report_lines.append(
                    f"  • {cat}: INR {pred_data['pred']:.2f} (Range: {low:.2f} - {high:.2f}) [Model: {pred_data['model']}]"
                )
            report_lines.append("")
        
        # Category Clustering Section
        if ml_insights['category_clusters']:
            report_lines.append("Semantic Category Clustering:")
            clusters_by_group = {}
            for cat, cluster_id in ml_insights['category_clusters'].items():
                if cluster_id not in clusters_by_group:
                    clusters_by_group[cluster_id] = []
                clusters_by_group[cluster_id].append(cat)
            for cluster_id in sorted(clusters_by_group.keys()):
                report_lines.append(f"  Cluster {cluster_id}: {', '.join(clusters_by_group[cluster_id])}")
            report_lines.append("")    
    else:
        report_lines.append("No financial data available for the given period.")
    
    return report_lines

# Main function to orchestrate the analysis
def analyze_financial_data(budget_data, from_date, to_date, range_data, all_data):
    # Prepare full history
    df_all, df_budget = prepare_data(all_data, budget_data)

    # Prepare range-filtered dataset
    df_range, _ = prepare_data(range_data, budget_data)
    df_range = filter_by_date(df_range, from_date, to_date)

    # Group by month/category using only range data
    month_category_data = group_by_month_category(df_range)
    
    return generate_report(
        df_budget, 
        month_category_data, 
        from_date, 
        to_date, 
        df_all   
    )       

def main():
    try:
        input_data = json.load(sys.stdin)
        
        budget_data = input_data.get('budgets', [])
        from_date = input_data.get('fromDate', '')
        to_date = input_data.get('toDate', '')
        
        range_data = input_data.get('rangeData', [])
        all_data = input_data.get('allData', [])
    
        # Perform financial data analysis
        report_lines = analyze_financial_data(budget_data, from_date, to_date, range_data, all_data)
        
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