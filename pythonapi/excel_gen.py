from io import BytesIO
import pandas as pd

def create_excel(result):
    output = BytesIO()

    # Convert summary and ML insights to DataFrames
    summary_df = pd.json_normalize(result['summary']['categories'])
    monthly_df = pd.json_normalize(result['summary']['monthly_breakdown'])

    anomalies_df = pd.DataFrame(result['ml_insights']['anomalies'])
    predictions_df = pd.DataFrame([
        {'category': k, **v} for k, v in result['ml_insights']['predictions']['items'].items()
    ])

    clusters_df = pd.DataFrame([
        {'cluster': cid, 'descriptions': ", ".join(desc)}
        for cid, desc in result['ml_insights']['description_clustering'].items()
    ])

    drift_data = result['ml_insights']['drift_report']['data']

    if isinstance(drift_data, dict):
        drift_df = pd.DataFrame([
            {'feature': k, **v} if isinstance(v, dict) else {'feature': k, 'value': v}
            for k, v in drift_data.items()
        ])
    elif isinstance(drift_data, list):
        drift_df = pd.DataFrame(drift_data)
    else:
        drift_df = pd.DataFrame()

    # Write to Excel inside memory buffer
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        summary_df.to_excel(writer, index=False, sheet_name='Summary')
        monthly_df.to_excel(writer, index=False, sheet_name='Monthly')
        anomalies_df.to_excel(writer, index=False, sheet_name='Anomalies')
        predictions_df.to_excel(writer, index=False, sheet_name='Predictions')
        clusters_df.to_excel(writer, index=False, sheet_name='Clusters')
        drift_df.to_excel(writer, index=False, sheet_name='Drift_Report')

    output.seek(0)
    return output