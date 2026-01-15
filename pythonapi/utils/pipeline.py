import sys
from .data_prep import DataPrepML
from services.anomaly import AnomalyML
from services.regresser import RegresserML
from services.cluster import ClusterML, normalize_description
from services.drift import DriftDetectionML

# ML-based financial analysis
def analyze_with_ml(budget_data, from_date, to_date, range_data, all_data):
     
    try: 
        # Base Transactions
        df_range, df_all = DataPrepML.build_base_transactions(range_data, all_data)

        # Monthly Aggregations
        monthly = DataPrepML.build_monthly_category(df_range)
        
        # Behavioral Features
        behavior = DataPrepML.build_monthly_behavioral_features(df_range)

        # Time Series Features
        monthly_ts = DataPrepML.get_timeseries_features(monthly)

        num_months = monthly_ts['month'].nunique()
        
        # Budget Metrics
        budget_metrics = DataPrepML.compute_budget_metrics(monthly_ts, budget_data, num_months)

        # ------------------ Anomaly Detection ------------------
        anomaly_features = DataPrepML.get_anomaly_features(df_range)
        anomaly_ml = monthly_ts[
            ['month', 'category', 'month_index', 'days_in_month', 'pct_share', 'amount']
        ].copy()
        anomaly_ml = anomaly_ml.merge(
            behavior[['month', 'category', 'weekend_ratio']],
            on=['month', 'category'],
            how='left'
        )
        anomaly_ml = anomaly_ml.join(
            anomaly_features[['roc_prev', 'volatility_3m']]
        )
        anomaly_ml = anomaly_ml.merge(
            budget_metrics[['category', 'budget_ratio']],
            on='category',
            how='left'
        )

        feature_cols = ['month_index','days_in_month','pct_share','weekend_ratio',
            'roc_prev','budget_ratio','volatility_3m','amount'
        ]
        anomaly_model = AnomalyML(model_dir='pythonapi/ml_models')
        anomaly_model.fit_unsupervised(anomaly_ml, feature_cols)
        anomaly_results = anomaly_model.detect_anomalies(anomaly_ml, feature_cols, return_scores=True)
        
        # ------------------ Regression ------------------
        ts_with_vol = monthly_ts.join(
            anomaly_features[['volatility_3m']]
        )
        regression_ml = ts_with_vol[
            [
                'month',
                'category',
                'amount',
                'lag_1',
                'lag_2',
                'lag_3',
                'volatility_3m',
                'month_of_year'
            ]
        ].copy()
        regression_model = RegresserML(model_dir='pythonapi/ml_models')
        regression_model.train_regressors(
            regression_ml,
            budgets_df=budget_metrics,
            n_lags=3
        )
        regression_results = regression_model.predict_next_month(regression_ml, budget_metrics)

        # ------------------ Clustering ------------------
        descriptions = (df_range['description'].dropna().astype(str).map(normalize_description).unique().tolist())

        if len(descriptions) > 1:
            # Create single instance for consistent state
            cluster_ml = ClusterML()
            
            # Step 1: Embed descriptions
            cluster_ml.fit_description_embeddings(descriptions)

            # Step 2: KMeans clustering
            clusters = cluster_ml.cluster_descriptions_kmeans()

            # Step 3: Group results
            clusters_by_group = {}
            for desc, cid in clusters.items():
                clusters_by_group.setdefault(cid, []).append(desc)

        # --------------------- Drift detection ---------------------
        drift_results = DriftDetectionML.compute_drift(monthly_ts, months_back=1)    

        # ------------------ Return structured output ------------------
        return {
            "features": monthly_ts,
            "anomaly_scores": anomaly_results,
            "regression_predictions": regression_results,
            "clusters": clusters_by_group,
            "drift_analysis": drift_results
        }
        
    except Exception as e:
        print(f"Warning: FinanceML analysis failed: {e}. Continuing with basic analysis.", file=sys.stderr)
        return {
            "features": None,
            "anomaly_scores": None,
            "regression_predictions": None,
            "clusters": {},
            "drift_analysis": None
        }
    