from io import BytesIO
import pandas as pd
import os

def create_excel(result):
    """Create Excel workbook from pipeline analysis results."""
    output = BytesIO()

    # Initialize workbook writer
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # ==================== FEATURES SHEET ====================
        if result.get('features') is not None:
            features_df = result['features'].copy()
            # Convert timestamp to string for Excel
            if 'month' in features_df.columns:
                features_df['month'] = features_df['month'].dt.strftime('%b-%Y')
            features_df.to_excel(writer, index=False, sheet_name='Features')
        
        # ==================== ANOMALIES SHEET ====================
        if result.get('anomaly_scores') is not None:
            anomalies = result['anomaly_scores']
            if isinstance(anomalies, list) and len(anomalies) > 0:
                anomalies_df = pd.DataFrame(anomalies)
                anomalies_df.to_excel(writer, index=False, sheet_name='Anomalies')
            else:
                # Create empty sheet if no anomalies
                pd.DataFrame({'message': ['No anomalies detected']}).to_excel(
                    writer, index=False, sheet_name='Anomalies'
                )
        
        # ==================== PREDICTIONS SHEET ====================
        if result.get('regression_predictions') is not None:
            predictions = result['regression_predictions']
            predictions_data = []
            for category, pred_info in predictions.items():
                pred_data = {
                    'category': category,
                    'predicted_amount': pred_info.get('pred', 0),
                    'confidence_low': pred_info.get('conf_int', (0, 0))[0],
                    'confidence_high': pred_info.get('conf_int', (0, 0))[1],
                    'model': pred_info.get('model', 'N/A'),
                    'model_score': pred_info.get('score', 0)
                }
                predictions_data.append(pred_data)
            
            if predictions_data:
                predictions_df = pd.DataFrame(predictions_data)
                predictions_df.to_excel(writer, index=False, sheet_name='Predictions')
            else:
                pd.DataFrame({'message': ['No predictions available']}).to_excel(
                    writer, index=False, sheet_name='Predictions'
                )
        
        # ==================== CLUSTERS SHEET ====================
        if result.get('clusters') is not None:
            clusters = result['clusters']
            clusters_data = []
            for cluster_id, descriptions in clusters.items():
                clusters_data.append({
                    'cluster_id': cluster_id,
                    'description_count': len(descriptions),
                    'descriptions': ", ".join(descriptions[:5]) + ("..." if len(descriptions) > 5 else "")
                })
            
            if clusters_data:
                clusters_df = pd.DataFrame(clusters_data)
                clusters_df.to_excel(writer, index=False, sheet_name='Clusters')
            else:
                pd.DataFrame({'message': ['No clusters generated']}).to_excel(
                    writer, index=False, sheet_name='Clusters'
                )
        
        # ==================== DRIFT ANALYSIS SHEET ====================
        if result.get('drift_analysis') is not None:
            drift = result['drift_analysis']
            
            if 'message' in drift:
                drift_df = pd.DataFrame({'status': [drift['message']]})
                drift_df.to_excel(writer, index=False, sheet_name='Drift Analysis')
            else:
                # Create drift summary
                drift_summary = {
                    'Metric': [
                        'Current Month',
                        'Previous Month',
                        'Jensen-Shannon Distance',
                        'Interpretation'
                    ],
                    'Value': [
                        drift.get('current_month', 'N/A'),
                        drift.get('previous_month', 'N/A'),
                        f"{drift.get('jensen_shannon', 0):.4f}",
                        _interpret_jensen_shannon(drift.get('jensen_shannon', 0))
                    ]
                }
                drift_summary_df = pd.DataFrame(drift_summary)
                
                # Write summary
                drift_summary_df.to_excel(writer, index=False, sheet_name='Drift Analysis', startrow=0)
                
                # Write top contributors
                startrow = len(drift_summary_df) + 3
                top_contributors = drift.get('top_psi_contributors', [])
                if top_contributors:
                    contrib_data = [
                        {'category': cat, 'psi_value': float(psi)}
                        for cat, psi in top_contributors
                    ]
                    contrib_df = pd.DataFrame(contrib_data)
                    contrib_df.to_excel(
                        writer,
                        index=False,
                        sheet_name='Drift Analysis',
                        startrow=startrow
                    )

        # ==================== VISUALIZATIONS SHEET ====================
        visualizations = result.get('visualizations', {})
        if visualizations:
            # Create a dedicated sheet for visualizations
            worksheet = workbook.add_worksheet('Visualizations')
            
            # Add title
            title_format = workbook.add_format({
                'bold': True,
                'font_size': 14,
                'font_color': '#2E86AB',
                'align': 'center'
            })
            worksheet.write(0, 0, 'Financial Analysis Visualizations', title_format)
            worksheet.set_column(0, 0, 60)  # Wide column for charts
            
            # Insert charts
            row = 2
            chart_order = [
                # Range data charts
                ('range_comparison', 'Spending by Category'),
                ('monthly_trends', 'Monthly Spending Trends'),
                ('category_breakdown', 'Category Distribution'),
                ('budget_vs_actual', 'Budget vs Actual Spending'),
                
                # Anomaly charts
                ('anomaly_scatter', 'Anomaly Detection - Scatter Plot'),
                ('anomaly_timeline', 'Anomaly Timeline'),
                ('anomaly_heatmap', 'Anomaly Heatmap'),
                
                # Regression charts
                ('predictions', 'Next Month Predictions'),
                ('prediction_confidence', 'Prediction Confidence Intervals'),
                ('model_performance', 'Model Performance'),
                
                # Clustering charts
                ('cluster_distribution', 'Cluster Distribution'),
                ('cluster_embeddings', 'Cluster Embeddings'),
                
                # Drift charts
                ('drift_timeline', 'Drift Timeline'),
                ('drift_contributors', 'Drift Contributors'),
                ('category_distribution_shift', 'Category Distribution Shift')
            ]
            
            for chart_key, chart_title in chart_order:
                chart_path = visualizations.get(chart_key)
                if chart_path and os.path.exists(chart_path):
                    # Add chart title
                    label_format = workbook.add_format({
                        'bold': True,
                        'font_size': 11,
                        'font_color': '#333333'
                    })
                    worksheet.write(row, 0, chart_title, label_format)
                    row += 1
                    
                    # Insert image
                    try:
                        worksheet.insert_image(row, 0, chart_path, {
                            'x_scale': 0.8,
                            'y_scale': 0.8
                        })
                        # Move to next chart position (approximate height)
                        row += 30
                    except Exception as e:
                        print(f"Warning: Could not embed chart {chart_key}: {e}")
                        row += 1

    output.seek(0)
    return output

def _interpret_jensen_shannon(js_value):
    """Interpret Jensen-Shannon distance value."""
    if js_value > 0.3:
        return "High drift - Significant spending pattern change"
    elif js_value > 0.1:
        return "Moderate drift - Some spending pattern changes"
    else:
        return "Low drift - Stable spending patterns"
