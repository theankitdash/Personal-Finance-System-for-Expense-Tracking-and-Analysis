import sys
import pandas as pd
import traceback
from .data_prep import DataPrepML
from ..services.anomaly import AnomalyML
from ..services.regresser import RegresserML
from ..services.cluster import ClusterML, normalize_description
from ..services.drift import DriftDetectionML
from ..services.visualizer import Visualizer


# ML-based financial analysis
def analyze_with_ml(budget_data, from_date, to_date, range_data, all_data):
     
    try: 
        # Base Transactions
        # We use all_data for robust model training (full history)
        df_range_processed, df_all = DataPrepML.build_base_transactions(range_data, all_data)

        if df_all.empty:
            return {
                "features": None,
                "anomaly_scores": None,
                "regression_predictions": None,
                "clusters": {},
                "drift_analysis": None
            }

        # Monthly Aggregations (Full History)
        monthly_all = DataPrepML.build_monthly_category(df_all)
        
        # Behavioral Features (Full History)
        behavior_all = DataPrepML.build_monthly_behavioral_features(df_all)

        # Time Series Features (Full History)
        monthly_ts_all = DataPrepML.get_timeseries_features(monthly_all)

        num_months_all = monthly_ts_all['month'].nunique()
        
        # Budget Metrics (Full History context)
        budget_metrics_all = DataPrepML.compute_budget_metrics(monthly_ts_all, budget_data, num_months_all)

        # ------------------ Anomaly Detection ------------------
        # Compute anomaly features on the AGGREGATED monthly data, not raw transactions
        anomaly_features_all = DataPrepML.get_anomaly_features(monthly_all)
        
        anomaly_ml = monthly_ts_all[
            ['month', 'category', 'month_index', 'days_in_month', 'pct_share', 'amount']
        ].copy()
        
        anomaly_ml = anomaly_ml.merge(
            behavior_all[['month', 'category', 'weekend_ratio']],
            on=['month', 'category'],
            how='left'
        )
        
        # Join anomaly features (indices align because both derived from monthly_all)
        anomaly_ml = anomaly_ml.join(
            anomaly_features_all[['roc_prev', 'volatility_3m']]
        )
        
        anomaly_ml = anomaly_ml.merge(
            budget_metrics_all[['category', 'budget_ratio']],
            on='category',
            how='left'
        )

        feature_cols = ['month_index','days_in_month','pct_share','weekend_ratio',
            'roc_prev','budget_ratio','volatility_3m','amount'
        ]
        
        anomaly_model = AnomalyML()
        anomaly_model.fit_unsupervised(anomaly_ml, feature_cols)
        
        # Detect on all data
        anomaly_results_all = anomaly_model.detect_anomalies(anomaly_ml, feature_cols, return_scores=True)
        
        # ------------------ Regression ------------------
        ts_with_vol = monthly_ts_all.join(
            anomaly_features_all[['volatility_3m']]
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
        
        regression_model = RegresserML()
        regression_model.train_regressors(
            regression_ml,
            budgets_df=budget_metrics_all,
            n_lags=3
        )
        # Predict next month
        regression_results = regression_model.predict_next_month(regression_ml, budget_metrics_all)

        # ------------------ Clustering ------------------
        # Clustering is specific to the descriptions in the VIEW
        descriptions = (df_range_processed['description'].dropna().astype(str).map(normalize_description).unique().tolist())

        clusters_by_group = {}
        if len(descriptions) > 1:
            cluster_ml = ClusterML()
            cluster_ml.fit_description_embeddings(descriptions)
            clusters = cluster_ml.cluster_descriptions_kmeans()
            
            for desc, cid in clusters.items():
                clusters_by_group.setdefault(cid, []).append(desc)

        # --------------------- Drift detection ---------------------
        drift_data = monthly_ts_all[monthly_ts_all['month'] <= pd.Timestamp(to_date)]
        drift_results = DriftDetectionML.compute_drift(drift_data, months_back=1)
        drift_history = DriftDetectionML.compute_drift_history(drift_data, max_periods=6)

        # ------------------ Filter Results for Return ------------------
        f_date = pd.to_datetime(from_date)
        t_date = pd.to_datetime(to_date)
        
        # Filter to include relevant months (month start date <= t_date and month end >= f_date?)
        # As 'month' is the 1st of the month:
        # We want months that fall into the range.
        # Simple Logic: Keep months appearing in the range. 
        # But 'month' column is a Timestamp.
        
        mask_range = (anomaly_results_all['month'] >= f_date.replace(day=1)) & \
                     (anomaly_results_all['month'] <= t_date)
                       
        final_anomaly_scores = anomaly_results_all[mask_range].copy()
        final_features = monthly_ts_all[mask_range].copy()

        # ------------------ Generate Visualizations ------------------
        visualizer = Visualizer()
        
        try:
            # Range data visualizations
            if not df_range_processed.empty:
                visualizer.plot_range_comparison(df_range_processed)
                if 'month' in final_features.columns:
                    visualizer.plot_monthly_trends(final_features)
                visualizer.plot_category_breakdown(df_range_processed)
            
            # Budget vs actual
            if not budget_metrics_all.empty:
                visualizer.plot_budget_vs_actual(budget_metrics_all)
            
            # Anomaly visualizations
            if not final_anomaly_scores.empty:
                visualizer.plot_anomaly_scatter(final_anomaly_scores)
                visualizer.plot_anomaly_timeline(final_anomaly_scores)
                visualizer.plot_anomaly_heatmap(final_anomaly_scores)
            
            # Regression visualizations
            if regression_results:
                visualizer.plot_predictions(regression_results, regression_ml)
                visualizer.plot_prediction_confidence(regression_results)
                visualizer.plot_model_performance(regression_results)
            
            # Clustering visualizations
            if clusters_by_group:
                visualizer.plot_cluster_distribution(clusters_by_group)
                # Get embeddings for 2D plot
                if 'cluster_ml' in locals():
                    embeddings, labels = cluster_ml.get_embeddings_and_labels()
                    if embeddings is not None:
                        visualizer.plot_cluster_embeddings(embeddings, labels)
            
            # Drift visualizations
            if drift_results and 'message' not in drift_results:
                visualizer.plot_drift_contributors(drift_results)
                visualizer.plot_category_distribution_shift(monthly_ts_all, drift_results)
            
            if drift_history:
                visualizer.plot_drift_timeline(drift_history)
            
            chart_paths = visualizer.get_all_chart_paths()
            
        except Exception as viz_error:
            print(f"Warning: Visualization generation failed: {viz_error}", file=sys.stderr)
            traceback.print_exc()
            chart_paths = {}

        return {
            "features": final_features,
            "anomaly_scores": final_anomaly_scores,
            "regression_predictions": regression_results,
            "clusters": clusters_by_group,
            "drift_analysis": drift_results,
            "drift_history": drift_history,
            "visualizations": chart_paths
        }
        
    except Exception as e:
        print(f"Warning: FinanceML analysis failed: {e}. Continuing with basic analysis.", file=sys.stderr)
        traceback.print_exc()
        return {
            "features": None,
            "anomaly_scores": None,
            "regression_predictions": None,
            "clusters": {},
            "drift_analysis": None,
            "drift_history": None,
            "visualizations": {}
        }
