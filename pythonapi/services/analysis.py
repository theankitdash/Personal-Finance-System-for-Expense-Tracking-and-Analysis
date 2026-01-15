import sys
import pandas as pd
from datetime import timedelta, datetime
from services.ml_module import FinanceML
from services.cluster import ClusterML, normalize_description

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
def analyze_with_ml(df_range, df_all, df_budget):
    
    ml_insights = {
        'anomalies': [],
        'predictions': {},
        'description_clustering': {},
        'drift_report': None
    }
    
    try:
        # Initialize FinanceML
        fm = FinanceML(model_dir='ml_models')
        clustergrp = ClusterML()
        
        # Build features from the data
        features_df_all = fm.build_features(df_all, df_budget)
        
        if features_df_all.empty:
            return ml_insights
        
        # 1. Fit unsupervised anomaly detection models
        fm.fit_unsupervised(features_df_all)
        
        # Detect anomalies on RANGE data (within selected period)
        features_df_range = fm.build_features(df_range, df_budget)
        anomalies_df = fm.detect_anomalies(features_df_range, return_scores=True)
        anomalies_df = anomalies_df[anomalies_df['is_anomaly']]      
        
        for _, row in anomalies_df.iterrows():
            ml_insights['anomalies'].append({
                'month': row['month_str'],
                'category': row['category'],
                'amount': float(row['amount']),
                'anomaly_votes': int(row['anomaly_votes']),
                'reason': _get_anomaly_reason(row),
                'lof_score': float(row['lof_score']) if pd.notna(row['lof_score']) else None,
                'ocsvm_score': float(row['ocsvm_score']) if pd.notna(row['ocsvm_score']) else None,
                'ae_mse': float(row['ae_mse']) if pd.notna(row['ae_mse']) else None
            })
        
        # 2. Compute drift detection
        drift_result = fm.compute_drift(df_all)
        if 'message' not in drift_result:
            ml_insights['drift_report'] = {
                'current_month': str(drift_result['current_month'].date()) if hasattr(drift_result['current_month'], 'date') else str(drift_result['current_month']),
                'previous_month': str(drift_result['previous_month'].date()) if hasattr(drift_result['previous_month'], 'date') else str(drift_result['previous_month']),
                'jensen_shannon': float(drift_result['jensen_shannon']),
                'drift_level': 'High - Significant spending pattern change' if drift_result['jensen_shannon'] > 0.3 else 'Moderate - Some spending pattern changes' if drift_result['jensen_shannon'] > 0.1 else 'Low - Stable spending patterns',
                'top_drifting_categories': [{'category': cat, 'change_intensity': float(psi), 'change_description': _get_change_description(psi)} for cat, psi in drift_result['top_psi_contributors'][:5]]
            }
        
        # 3. Train regressors and predict next month
        fm.train_regressors(df_all, df_budget)
        predictions = fm.predict_next_month(df_all, df_budget)
        
        for cat, pred_data in predictions.items():
            low, high = pred_data['conf_int']
            ml_insights['predictions'][cat] = {
                'predicted_amount': float(pred_data['pred']),
                'confidence_low': float(low),
                'confidence_high': float(high),
                'confidence_range_description': f"₹{float(low):,.0f} - ₹{float(high):,.0f}",
                'model': pred_data['model'],
                'model_accuracy': float(pred_data['score']),
                'accuracy_label': _get_accuracy_label(pred_data['score'])
            }
        
        # 4. Semantic description clustering 
        try:
            descriptions = (df_range['description'].dropna().astype(str).map(normalize_description).unique().tolist())

            if len(descriptions) > 1:
                # Step 1: Embed descriptions
                clustergrp.fit_description_embeddings(descriptions)

                # Step 2: KMeans clustering
                clusters = clustergrp.cluster_descriptions_kmeans()

                # Step 3: Group results
                clusters_by_group = {}
                for desc, cid in clusters.items():
                    clusters_by_group.setdefault(cid, []).append(desc)

                ml_insights['description_clusters'] = {
                    f'group_{cid + 1}': descs
                    for cid, descs in clusters_by_group.items()
                }
        except Exception as e:
            print("Description clustering failed:", e) 
    
    except Exception as e:
        print(f"Warning: FinanceML analysis failed: {e}. Continuing with basic analysis.", file=sys.stderr)
    
    return ml_insights

# Helper functions for user-friendly descriptions
def _get_anomaly_reason(row):
    """Generate a user-friendly description of why this is an anomaly"""
    votes = int(row['anomaly_votes'])
    if votes >= 2:
        return "Unusual spending detected - significantly different from your normal pattern"
    return "Slightly unusual spending - worth reviewing"

def _get_change_description(psi_value):
    """Convert PSI value to user-friendly change description"""
    if psi_value > 0.5:
        return "Very significant change"
    elif psi_value > 0.3:
        return "Significant change"
    elif psi_value > 0.1:
        return "Moderate change"
    else:
        return "Minor change"

def _get_accuracy_label(score):
    """Convert model accuracy score to user-friendly label"""
    if score > 0.85:
        return "Very accurate"
    elif score > 0.70:
        return "Good accuracy"
    elif score > 0.50:
        return "Fair accuracy"
    else:
        return "Low accuracy"

# Build summary report data with user-friendly formatting
def build_summary_report(df_budget, month_category_data, from_date, to_date, df_expenses):
    # Ensure budget amounts are floats
    df_budget = df_budget.copy()
    df_budget['amount'] = pd.to_numeric(df_budget['amount'], errors='coerce').fillna(0.0)

    # Calculate number of months in the period
    from_dt = pd.to_datetime(from_date)
    to_dt = pd.to_datetime(to_date)
    num_months = (to_dt.year - from_dt.year) * 12 + (to_dt.month - from_dt.month) + 1
    num_months = max(1, num_months)  # At least 1 month
    
    # Calculate total budget for the period (monthly budget * number of months)
    monthly_budget_total = float(df_budget['amount'].sum()) if not df_budget.empty else 0.0
    period_budget_total = monthly_budget_total * num_months
    
    summary = {
        'period': {
            'from': from_date,
            'to': to_date,
            'display': f"{from_date} to {to_date}",
            'num_months': num_months
        },
        'total_spending': 0.0,
        'total_spending_display': "₹0",
        'total_budget': period_budget_total,
        'total_budget_display': f"₹{period_budget_total:,.0f}" if period_budget_total > 0 else "₹0",
        'monthly_budget': monthly_budget_total,
        'monthly_budget_display': f"₹{monthly_budget_total:,.0f}" if monthly_budget_total > 0 else "₹0",
        'categories': [],
        'monthly_breakdown': [],
        'current_month': None,
        'overall_health': None
    }
    
    if not month_category_data.empty:
        current_month = datetime.now().strftime('%b-%Y')
        
        # Category-wise breakdown with user-friendly formatting
        for category in month_category_data.columns:
            total_cat = month_category_data[category].sum()
            budget_cat = df_budget[df_budget['category'] == category]['amount'].values
            monthly_budget_cat = float(budget_cat[0]) if len(budget_cat) > 0 else 0.0
            period_budget_cat = monthly_budget_cat * num_months
            
            status = "WITHIN"
            variance = 0.0
            status_emoji = "✓"
            if period_budget_cat > 0:
                variance = ((total_cat - period_budget_cat) / period_budget_cat) * 100
                status = "OVER" if variance > 0 else "WITHIN"
                status_emoji = "⚠" if variance > 0 else "✓"
            
            summary['categories'].append({
                'name': category,
                'total_spent': float(total_cat),
                'total_spent_display': f"₹{float(total_cat):,.0f}",
                'budget': period_budget_cat,
                'budget_display': f"₹{period_budget_cat:,.0f}" if period_budget_cat > 0 else "No budget set",
                'monthly_budget': monthly_budget_cat,
                'monthly_budget_display': f"₹{monthly_budget_cat:,.0f}" if monthly_budget_cat > 0 else "No budget set",
                'variance_percent': float(variance),
                'variance_display': f"{variance:+.1f}%",
                'status': status,
                'status_emoji': status_emoji,
                'status_message': f"Over budget by {abs(variance):.1f}%" if variance > 0 else f"Within budget by {abs(variance):.1f}%"
            })
        
        # Monthly breakdown
        for month, row in month_category_data.iterrows():
            month_total = row.sum()
            summary['monthly_breakdown'].append({
                'month': month,
                'total': float(month_total),
                'total_display': f"₹{float(month_total):,.0f}",
                'budget': monthly_budget_total,
                'budget_display': f"₹{monthly_budget_total:,.0f}",
                'categories': {cat: f"₹{float(amount):,.0f}" for cat, amount in row.items()}
            })
        
        summary['total_spending'] = float(month_category_data.sum().sum())
        summary['total_spending_display'] = f"₹{summary['total_spending']:,.0f}"

        # Overall health assessment based on entire period
        period_spent = summary['total_spending']
        period_budget = summary['total_budget']
        overall_usage_percent = (period_spent / period_budget * 100) if period_budget > 0 else 0
        
        # Health assessment
        health = "Good - On track with budget"
        if overall_usage_percent > 90:
            health = "Caution - Nearing budget limit"
        if overall_usage_percent > 100:
            health = "Alert - Over budget"
        
        summary['overall_health'] = health
        summary['overall_usage_percent'] = float(overall_usage_percent)
        summary['overall_usage_label'] = f"{overall_usage_percent:.1f}%"
        
        # Current month data with health assessment
        if current_month in month_category_data.index:
            current_spent = month_category_data.loc[current_month].sum()
            current_budget = summary['total_budget']
            usage_percent = (current_spent / current_budget * 100) if current_budget > 0 else 0
            
            summary['current_month'] = {
                'month': current_month,
                'spent': float(current_spent),
                'spent_display': f"₹{float(current_spent):,.0f}",
                'budget': float(current_budget),
                'budget_display': f"₹{float(current_budget):,.0f}",
                'remaining': float(current_budget - current_spent),
                'remaining_display': f"₹{float(current_budget - current_spent):,.0f}",
                'usage_percent': float(usage_percent),
                'usage_label': f"{usage_percent:.1f}%",
                'health_status': "Good - On track" if usage_percent <= 90 else "Caution - Nearing limit" if usage_percent <= 100 else "Alert - Over budget"
            }

    return summary

# Main function to orchestrate the analysis
def analyze_financial_data(budget_data, from_date, to_date, range_data, all_data):
    """
    Returns a comprehensive JSON response with:
    - Summary statistics with user-friendly formatting
    - Detailed monthly breakdowns
    - ML-based insights (anomalies, drift, predictions, clustering)
    """
    try:
        # Prepare full history
        df_all, df_budget = prepare_data(all_data, budget_data)

        # Prepare range-filtered dataset
        df_range, df_budget = prepare_data(range_data, budget_data)

        df_range = filter_by_date(df_range, from_date, to_date)

        # Group by month/category using only range data
        month_category_data = group_by_month_category(df_range)
        
        # Build summary report
        summary = build_summary_report(df_budget, month_category_data, from_date, to_date, df_all)
        
        # Run ML analysis
        ml_insights = analyze_with_ml(df_range, df_all, df_budget)
        
        # Return comprehensive JSON response with user-friendly formatting
        return {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'message': f"Analysis complete for period {from_date} to {to_date}",
            'summary': summary,
            'ml_insights': {
                'anomalies': {
                    'count': len(ml_insights['anomalies']),
                    'items': ml_insights['anomalies'],
                    'description': f"Found {len(ml_insights['anomalies'])} unusual spending patterns" if ml_insights['anomalies'] else "No unusual spending patterns detected"
                },
                'drift_report': {
                    'data': ml_insights['drift_report'],
                    'description': "Analysis of how your spending patterns have changed" if ml_insights['drift_report'] else "Insufficient data for drift analysis"
                },
                'predictions': {
                    'count': len(ml_insights['predictions']),
                    'items': ml_insights['predictions'],
                    'description': "Next month spending predictions for each category" if ml_insights['predictions'] else "Unable to generate predictions"
                },
                'description_clustering': ml_insights.get('description_clusters', {})
            }
        }
    
    except Exception as e:
        print(f"Error in analyze_financial_data: {e}", file=sys.stderr)
        return {
            'status': 'error',
            'message': f"Unable to complete analysis: {str(e)}",
            'timestamp': datetime.now().isoformat()
        }