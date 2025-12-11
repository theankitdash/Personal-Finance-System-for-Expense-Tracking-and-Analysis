from io import BytesIO
import pandas as pd

def create_excel(result):
    output = BytesIO()

    # Convert summary and ML insights to DataFrames
    summary_df = pd.json_normalize(result['summary']['categories'])
    monthly_df = pd.json_normalize(result['summary']['monthly_breakdown'])

    anomalies_df = pd.DataFrame(result['ml_insights']['anomalies'])
    predictions_df = pd.DataFrame([
        {'category': k, **v} for k, v in result['ml_insights']['predictions'].items()
    ])

    clusters_df = pd.DataFrame([
        {'cluster': cid, 'categories': ", ".join(cats)}
        for cid, cats in result['ml_insights']['category_clustering'].items()
    ])

    # Write to Excel inside memory buffer
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        summary_df.to_excel(writer, index=False, sheet_name='Summary')
        monthly_df.to_excel(writer, index=False, sheet_name='Monthly')
        anomalies_df.to_excel(writer, index=False, sheet_name='Anomalies')
        predictions_df.to_excel(writer, index=False, sheet_name='Predictions')
        clusters_df.to_excel(writer, index=False, sheet_name='Clusters')

    output.seek(0)
    return output