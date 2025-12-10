import sys
import pandas as pd
from datetime import timedelta, datetime
from pythonapi.ml_module import FinanceML

# Function to convert and prepare data for analysis
def prepare_data(expenses, budget_data):
    df_expenses = pd.DataFrame(expenses)
    df_budget = pd.DataFrame(budget_data)

     # Convert all columns to numeric where possible
    for col in df_expenses.columns:
        if col not in ['category', 'description', 'date', 'month']:
            try:
                df_expenses[col] = pd.to_numeric(df_expenses[col])
            except (ValueError, TypeError):
                pass  # Keep original value if conversion fails
    
    for col in df_budget.columns:
        if col not in ['category', 'description']:
            try:
                df_budget[col] = pd.to_numeric(df_budget[col])
            except (ValueError, TypeError):
                pass  # Keep original value if conversion fails

    # Process date column and create month column
    df_expenses['date'] = pd.to_datetime(df_expenses['date']) + timedelta(days=1)
    df_expenses['month'] = df_expenses['date'].dt.strftime('%b-%Y')
    
    return df_expenses, df_budget

# Function to filter data by date range
def filter_by_date(df_expenses, from_date, to_date):
    return df_expenses[(df_expenses['date'] >= from_date) & (df_expenses['date'] <= to_date)]

# Function to group data by month and category
def group_by_month_category(df_expenses):
    try:
        month_category_data = df_expenses.groupby(['month', 'category'])['amount'].sum().unstack(fill_value=0)
    except KeyError as e:
        print(f"Missing column in aggregated data: {e}", file=sys.stderr)
        month_category_data = pd.DataFrame()
    
    return month_category_data

# ML-based financial analysis using FinanceML module
def analyze_with_ml(df_expenses, df_budget):
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
        'predictions': {},
        'category_clusters': {}
    }
    
    try:
        # Convert Decimal types to float
        df_aggregated_clean = df_expenses.copy()
        for col in df_aggregated_clean.select_dtypes(include=['object']).columns:
            try:
                df_aggregated_clean[col] = pd.to_numeric(df_aggregated_clean[col])
            except (ValueError, TypeError):
                pass
        
        df_budget_clean = df_budget.copy()
        for col in df_budget_clean.select_dtypes(include=['object']).columns:
            try:
                df_budget_clean[col] = pd.to_numeric(df_budget_clean[col])
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
        drift_result = fm.compute_drift(df_aggregated_clean)
        if 'message' not in drift_result:
            ml_insights['drift_report'] = {
                'current_month': str(drift_result['current_month'].date()) if hasattr(drift_result['current_month'], 'date') else str(drift_result['current_month']),
                'previous_month': str(drift_result['previous_month'].date()) if hasattr(drift_result['previous_month'], 'date') else str(drift_result['previous_month']),
                'jensen_shannon': float(drift_result['jensen_shannon']),
                'interpretation': 'HIGH' if drift_result['jensen_shannon'] > 0.3 else 'MODERATE' if drift_result['jensen_shannon'] > 0.1 else 'LOW',
                'top_drifting_categories': [{'category': cat, 'psi': float(psi)} for cat, psi in drift_result['top_psi_contributors'][:5]]
            }
        
        # 3. Train regressors and predict next month
        fm.train_regressors(df_aggregated_clean, df_budget_clean)
        predictions = fm.predict_next_month(df_aggregated_clean, df_budget_clean)
        
        for cat, pred_data in predictions.items():
            low, high = pred_data['conf_int']
            ml_insights['predictions'][cat] = {
                'predicted_amount': float(pred_data['pred']),
                'confidence_low': float(low),
                'confidence_high': float(high),
                'model': pred_data['model'],
                'model_accuracy': float(pred_data['score'])
            }
        
        # 4. Semantic category clustering (optional)
        try:
            categories = df_aggregated_clean['category'].unique().tolist()
            if len(categories) > 1:
                fm.fit_category_embeddings(categories)
                n_clusters = max(2, min(len(categories) // 3, 8))
                clusters = fm.cluster_categories(n_clusters=n_clusters)
                clusters_by_group = {}
                for cat, cluster_id in clusters.items():
                    if cluster_id not in clusters_by_group:
                        clusters_by_group[cluster_id] = []
                    clusters_by_group[cluster_id].append(cat)
                ml_insights['category_clusters'] = {f'cluster_{cid}': cats for cid, cats in clusters_by_group.items()}
        except Exception as e:
            pass  # Category clustering is optional
    
    except Exception as e:
        print(f"Warning: FinanceML analysis failed: {e}. Continuing with basic analysis.", file=sys.stderr)
    
    return ml_insights

# Build summary report data
def build_summary_report(df_budget, month_category_data, from_date, to_date, df_expenses):
    # Ensure budget amounts are floats
    df_budget = df_budget.copy()
    df_budget['amount'] = pd.to_numeric(df_budget['amount'], errors='coerce').fillna(0.0)
    
    summary = {
        'period': {
            'from': from_date,
            'to': to_date
        },
        'total_spending': 0.0,
        'total_budget': float(df_budget['amount'].sum()) if not df_budget.empty else 0.0,
        'categories': [],
        'monthly_breakdown': [],
        'current_month': None
    }
    
    if not month_category_data.empty:
        current_month = datetime.now().strftime('%b-%Y')
        
        # Category-wise breakdown
        for category in month_category_data.columns:
            total_cat = month_category_data[category].sum()
            budget_cat = df_budget[df_budget['category'] == category]['amount'].values
            budget_amount = float(budget_cat[0]) if len(budget_cat) > 0 else 0.0
            
            status = "WITHIN"
            variance = 0.0
            if budget_amount > 0:
                variance = ((total_cat - budget_amount) / budget_amount) * 100
                status = "OVER" if variance > 0 else "WITHIN"
            
            summary['categories'].append({
                'name': category,
                'total_spent': float(total_cat),
                'budget': budget_amount,
                'variance_percent': float(variance),
                'status': status
            })
        
        # Monthly breakdown
        for month, row in month_category_data.iterrows():
            month_total = row.sum()
            summary['monthly_breakdown'].append({
                'month': month,
                'total': float(month_total),
                'categories': {cat: float(amount) for cat, amount in row.items()}
            })
        
        summary['total_spending'] = float(month_category_data.sum().sum())
        
        # Current month data
        if current_month in month_category_data.index:
            current_spent = month_category_data.loc[current_month].sum()
            current_budget = summary['total_budget']
            summary['current_month'] = {
                'month': current_month,
                'spent': float(current_spent),
                'budget': float(current_budget),
                'remaining': float(current_budget - current_spent),
                'usage_percent': float((current_spent / current_budget * 100) if current_budget > 0 else 0)
            }
    
    return summary

# Main function to orchestrate the analysis
def analyze_financial_data(budget_data, from_date, to_date, range_data, all_data):
    """
    Returns a comprehensive JSON response with:
    - Summary statistics
    - Detailed monthly breakdowns
    - ML-based insights (anomalies, drift, predictions, clustering)
    """
    try:
        # Prepare full history
        df_all, df_budget = prepare_data(all_data, budget_data)

        # Prepare range-filtered dataset
        df_range, _ = prepare_data(range_data, budget_data)
        df_range = filter_by_date(df_range, from_date, to_date)

        # Group by month/category using only range data
        month_category_data = group_by_month_category(df_range)
        
        # Build summary report
        summary = build_summary_report(df_budget, month_category_data, from_date, to_date, df_all)
        
        # Run ML analysis
        ml_insights = analyze_with_ml(df_all, df_budget)
        
        # Return comprehensive JSON response
        return {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'summary': summary,
            'ml_insights': {
                'anomalies': ml_insights['anomalies'],
                'drift_analysis': ml_insights['drift_report'],
                'predictions': ml_insights['predictions'],
                'category_clustering': ml_insights['category_clusters']
            }
        }
    
    except Exception as e:
        print(f"Error in analyze_financial_data: {e}", file=sys.stderr)
        return {
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }     