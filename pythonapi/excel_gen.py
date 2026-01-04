from io import BytesIO
import pandas as pd

def create_excel(result):
    """Create Excel workbook from pipeline analysis results."""
    output = BytesIO()

    # Initialize workbook writer
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # Create formats
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#4472C4',
            'font_color': 'white',
            'border': 1
        })
        
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