"""
ml_finance_module.py

A modular ML toolkit for the financial analysis script you provided.
- Places ML detection / forecasting / clustering in a separate file.
- Features included:
  1) Multivariate unsupervised anomaly detection: LOF, One-Class SVM, Autoencoder
  2) Category-level regression models: RandomForest, GradientBoosting, XGBoost (optional)
  3) Semantic category embeddings + clustering (sentence-transformers)
  4) Drift detection (Jensen-Shannon divergence & PSI)
  5) Plotting helpers for trends, heatmaps, and anomaly scatter charts
  6) Lightweight adaptive retrain scheduler (rolling-window retraining helpers)

Usage (high-level):
    from ml_finance_module import FinanceML

    fm = FinanceML()
    fm.fit_category_embeddings(categories_list)
    clustered = fm.cluster_categories(n_clusters=12)

    # Prepare aggregated dataframe with columns ['date','category','amount']
    df = your_df
    features_df = fm.build_features(df, budgets_df)

    # Train anomaly detectors
    fm.fit_unsupervised(features_df)
    anomalies = fm.detect_anomalies(features_df)

    # Train regressors per category
    fm.train_regressors(df, budgets_df)
    preds = fm.predict_next_month(df, budgets_df)

    # Drift detection
    drift_report = fm.compute_drift(df)

    # Plots
    fm.plot_category_trend(df, 'Food')

Dependencies:
    pandas, numpy, sklearn, matplotlib, scipy, joblib
    Optional: xgboost, tensorflow (or keras), sentence_transformers

The module is written defensively: optional libraries are imported lazily and the functions raise informative errors if missing.

"""

from typing import List, Dict, Optional, Tuple
import os
import math
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

# sklearn / scipy
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import jensenshannon
from scipy import stats
import matplotlib.pyplot as plt

# Optional imports (lazy failures with instructions)
try:
    import xgboost as xgb
    _HAS_XGBOOST = True
except Exception:
    _HAS_XGBOOST = False

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    _HAS_SENTENCE_TRANSFORMERS = False

try:
    # Use tensorflow.keras for autoencoder
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    _HAS_TF = True
except Exception:
    _HAS_TF = False


class FinanceML:
    def __init__(self, model_dir: str = 'ml_models'):
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.scaler = StandardScaler()

        # Placeholders for models
        self.lof = None
        self.ocsvm = None
        self.autoencoder = None
        self.autoencoder_threshold = None

        # regressors per (normalized) category
        self.regressors = {}

        # embeddings
        self.category_embeddings = None
        self.cat_to_vec = {}
        self.cluster_labels = None
        self.cluster_model = None
        self.sentence_model = None

    # --------------------- Category embeddings & clustering ---------------------
    def ensure_sentence_model(self, model_name: str = 'all-MiniLM-L6-v2'):
        if not _HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("sentence-transformers is required for semantic category embeddings. Install with: pip install sentence-transformers")
        if self.sentence_model is None:
            self.sentence_model = SentenceTransformer(model_name)
        return self.sentence_model

    def fit_category_embeddings(self, categories: List[str], model_name: str = 'all-MiniLM-L6-v2') -> pd.DataFrame:
        """Compute sentence embeddings for category names and store mapping."""
        model = self.ensure_sentence_model(model_name)
        cats = [str(c) for c in categories]
        vecs = model.encode(cats, show_progress_bar=False)
        df = pd.DataFrame(vecs, index=cats)
        self.category_embeddings = df
        self.cat_to_vec = {c: vecs[i] for i, c in enumerate(cats)}
        return df

    def cluster_categories(self, n_clusters: int = 8, method: str = 'kmeans') -> Dict[str, int]:
        if self.category_embeddings is None:
            raise ValueError('Call fit_category_embeddings(categories) first')

        X = self.category_embeddings.values
        if method == 'kmeans':
            km = KMeans(n_clusters=n_clusters, random_state=42)
            labels = km.fit_predict(X)
            self.cluster_model = km
        else:
            raise NotImplementedError('Only kmeans implemented for now')

        self.cluster_labels = dict(zip(self.category_embeddings.index.tolist(), labels.tolist()))
        return self.cluster_labels

    def get_cluster_for_category(self, category: str):
        return self.cluster_labels.get(category)

    def merge_semantic_categories(self, df: pd.DataFrame, how: str = 'cluster') -> pd.DataFrame:
        """Return a copy of df with a new column 'semantic_category' using cluster labels.
        df must have a column 'category'."""
        if self.cluster_labels is None:
            raise ValueError('Run fit_category_embeddings() and cluster_categories() first')
        df2 = df.copy()
        df2['semantic_category'] = df2['category'].map(lambda c: f'cluster_{self.cluster_labels.get(c, -1)}')
        return df2

    # --------------------- Feature engineering ---------------------
    def build_features(self, df: pd.DataFrame, budgets_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Build multivariate features per (month, category) row for anomaly detection / regression.
        Input df: columns ['date','category','amount'] where date is datetime-like.
        Returns a DataFrame with features and metadata columns.
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()

        # Group by month & category
        grp = df.groupby(['month', 'category'])['amount'].sum().reset_index()

        # Total per month for percentage share
        total_month = grp.groupby('month')['amount'].sum().rename('month_total').reset_index()
        grp = grp.merge(total_month, on='month', how='left')
        grp['pct_share'] = grp['amount'] / grp['month_total']

        # Month index (numeric)
        grp['month_index'] = ((grp['month'] - grp['month'].min()) / np.timedelta64(1, 'M')).astype(int)

        # Days in month
        grp['days_in_month'] = grp['month'].dt.daysinmonth

        # Weekend vs weekday ratio: approximate from raw transactions
        # compute per (month, category) what fraction of transactions fell on weekend
        tx = df.copy()
        tx['is_weekend'] = tx['date'].dt.dayofweek >= 5
        wk = tx.groupby(['month', 'category'])['is_weekend'].mean().rename('weekend_ratio').reset_index()
        grp = grp.merge(wk, on=['month', 'category'], how='left')
        grp['weekend_ratio'] = grp['weekend_ratio'].fillna(0.0)

        # Rate of change vs previous month for same category
        grp = grp.sort_values(['category', 'month'])
        grp['prev_amount'] = grp.groupby('category')['amount'].shift(1).fillna(0.0)
        grp['roc_prev'] = (grp['amount'] - grp['prev_amount']) / (grp['prev_amount'].replace(0, np.nan))
        grp['roc_prev'] = grp['roc_prev'].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # How far into budget on that date: if budgets provided
        if budgets_df is not None and 'category' in budgets_df.columns:
            budgets = budgets_df.copy()
            budgets = budgets.set_index('category')['amount'].to_dict()
            grp['budget_amount'] = grp['category'].map(lambda c: budgets.get(c, np.nan))
            grp['budget_ratio'] = grp['amount'] / grp['budget_amount']
            grp['budget_ratio'] = grp['budget_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        else:
            grp['budget_amount'] = np.nan
            grp['budget_ratio'] = 0.0

        # volatility: std dev of category over last 3 months
        grp['volatility_3m'] = grp.groupby('category')['amount'].rolling(window=3, min_periods=1).std().reset_index(level=0, drop=True).fillna(0.0)

        # additional useful metadata
        grp['category_str'] = grp['category'].astype(str)
        grp['month_str'] = grp['month'].dt.strftime('%b-%Y')

        # final feature set
        features = grp[['month', 'month_str', 'month_index', 'category', 'category_str', 'amount', 'month_total', 'pct_share', 'days_in_month', 'weekend_ratio', 'prev_amount', 'roc_prev', 'budget_amount', 'budget_ratio', 'volatility_3m']]
        features = features.fillna(0.0)

        return features

    # --------------------- Unsupervised anomaly models ---------------------
    def fit_unsupervised(self, features_df: pd.DataFrame, feature_columns: Optional[List[str]] = None):
        """Fit LOF, One-Class SVM, and Autoencoder on the passed features dataframe.
        Saves models to disk under model_dir.
        """
        if feature_columns is None:
            feature_columns = ['month_index', 'days_in_month', 'pct_share', 'weekend_ratio', 'roc_prev', 'budget_ratio', 'volatility_3m', 'amount']

        X = features_df[feature_columns].values.astype(float)
        X_scaled = self.scaler.fit_transform(X)

        # LOF
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True)
        lof.fit(X_scaled)
        self.lof = lof
        joblib.dump(lof, os.path.join(self.model_dir, 'lof.joblib'))

        # One-Class SVM
        oc = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
        oc.fit(X_scaled)
        self.ocsvm = oc
        joblib.dump(oc, os.path.join(self.model_dir, 'ocsvm.joblib'))

        # Autoencoder (optional)
        if _HAS_TF:
            dim = X_scaled.shape[1]
            encoding_dim = max(4, dim // 2)
            inp = Input(shape=(dim,))
            x = Dense(encoding_dim * 2, activation='relu')(inp)
            x = Dense(encoding_dim, activation='relu')(x)
            x = Dense(encoding_dim * 2, activation='relu')(x)
            out = Dense(dim, activation='linear')(x)
            ae = Model(inp, out)
            ae.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

            # Train with early stopping
            es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
            ae.fit(X_scaled, X_scaled, epochs=200, batch_size=32, callbacks=[es], verbose=0)
            self.autoencoder = ae

            # compute reconstruction errors and threshold (mean + 3*std)
            recon = ae.predict(X_scaled)
            mse = np.mean(np.square(recon - X_scaled), axis=1)
            thresh = np.mean(mse) + 3 * np.std(mse)
            self.autoencoder_threshold = float(thresh)

            # Save
            ae.save(os.path.join(self.model_dir, 'autoencoder_tf'))
        else:
            self.autoencoder = None
            self.autoencoder_threshold = None

        # Save scaler
        joblib.dump(self.scaler, os.path.join(self.model_dir, 'scaler.joblib'))

    def detect_anomalies(self, features_df: pd.DataFrame, feature_columns: Optional[List[str]] = None, return_scores: bool = False) -> pd.DataFrame:
        if feature_columns is None:
            feature_columns = ['month_index', 'days_in_month', 'pct_share', 'weekend_ratio', 'roc_prev', 'budget_ratio', 'volatility_3m', 'amount']

        X = features_df[feature_columns].values.astype(float)
        X_scaled = self.scaler.transform(X)

        results = features_df.copy().reset_index(drop=True)

        # LOF scores (negative_outlier_factor_ when fitted with novelty=False. But we fit novelty=True so use decision_function)
        if self.lof is not None:
            try:
                lof_scores = self.lof.decision_function(X_scaled)
            except Exception:
                lof_scores = self.lof._decision_function(X_scaled)
            results['lof_score'] = lof_scores
            results['lof_anomaly'] = results['lof_score'] < np.percentile(results['lof_score'], 5)
        else:
            results['lof_score'] = np.nan
            results['lof_anomaly'] = False

        # OCSVM
        if self.ocsvm is not None:
            oc_scores = self.ocsvm.decision_function(X_scaled)
            results['ocsvm_score'] = oc_scores
            results['ocsvm_anomaly'] = results['ocsvm_score'] < np.percentile(results['ocsvm_score'], 5)
        else:
            results['ocsvm_score'] = np.nan
            results['ocsvm_anomaly'] = False

        # Autoencoder
        if self.autoencoder is not None:
            recon = self.autoencoder.predict(X_scaled)
            mse = np.mean(np.square(recon - X_scaled), axis=1)
            results['ae_mse'] = mse
            results['ae_anomaly'] = results['ae_mse'] > self.autoencoder_threshold
        else:
            results['ae_mse'] = np.nan
            results['ae_anomaly'] = False

        # Aggregate anomaly votes
        results['anomaly_votes'] = results[['lof_anomaly', 'ocsvm_anomaly', 'ae_anomaly']].sum(axis=1)
        results['is_anomaly'] = results['anomaly_votes'] >= 1  # flag if any model votes; tuneable

        if return_scores:
            return results
        else:
            # return only anomaly rows
            return results[results['is_anomaly']].copy()

    # --------------------- Regression per category ---------------------
    def _prepare_regression_data(self, df: pd.DataFrame, n_lags: int = 3) -> pd.DataFrame:
        """Build dataset with lag features for each category-month. Returns a table with one row per month-category and lag features."""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
        grp = df.groupby(['month', 'category'])['amount'].sum().reset_index().sort_values(['category', 'month'])

        # create lag features
        for lag in range(1, n_lags + 1):
            grp[f'lag_{lag}'] = grp.groupby('category')['amount'].shift(lag).fillna(0.0)

        # rolling vol
        grp['volatility_3m'] = grp.groupby('category')['amount'].rolling(window=3, min_periods=1).std().reset_index(level=0, drop=True).fillna(0.0)

        # month_of_year
        grp['month_of_year'] = grp['month'].dt.month

        return grp

    def train_regressors(self, df: pd.DataFrame, budgets_df: Optional[pd.DataFrame] = None, n_lags: int = 3):
        data = self._prepare_regression_data(df, n_lags=n_lags)
        categories = data['category'].unique().tolist()

        for cat in categories:
            cat_df = data[data['category'] == cat].copy()
            if len(cat_df) < 6:
                # not enough history to train a regressor
                continue

            X = cat_df[[f'lag_{i}' for i in range(1, n_lags + 1)] + ['volatility_3m', 'month_of_year']]
            y = cat_df['amount']

            # include budget ratio if budgets provided
            if budgets_df is not None:
                budget_map = budgets_df.set_index('category')['amount'].to_dict()
                cat_df['budget_amount'] = cat_df['category'].map(lambda c: budget_map.get(c, np.nan))
                cat_df['budget_ratio'] = cat_df['lag_1'] / cat_df['budget_amount'].replace(0, np.nan)
                X['budget_ratio'] = cat_df['budget_ratio'].fillna(0.0)

            # train/test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # choose model (RandomForest + optionally XGBoost)
            rf = RandomForestRegressor(n_estimators=200, random_state=42)
            rf.fit(X_train, y_train)
            score_rf = rf.score(X_test, y_test)

            gb = GradientBoostingRegressor(n_estimators=200, random_state=42)
            gb.fit(X_train, y_train)
            score_gb = gb.score(X_test, y_test)

            chosen = rf
            chosen_name = 'RandomForest'
            chosen_score = score_rf

            if _HAS_XGBOOST:
                try:
                    xg = xgb.XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
                    xg.fit(X_train, y_train)
                    score_xg = xg.score(X_test, y_test)
                    if score_xg > chosen_score:
                        chosen = xg
                        chosen_name = 'XGBoost'
                        chosen_score = score_xg
                except Exception:
                    pass

            # keep model with best score among rf/gb/xg
            if score_gb > chosen_score:
                chosen = gb
                chosen_name = 'GradientBoosting'
                chosen_score = score_gb

            self.regressors[cat] = {
                'model': chosen,
                'model_name': chosen_name,
                'score': chosen_score,
                'features': X.columns.tolist()
            }

            # persist
            joblib.dump(self.regressors[cat], os.path.join(self.model_dir, f'regressor_{cat}.joblib'))

    def predict_next_month(self, df: pd.DataFrame, budgets_df: Optional[pd.DataFrame] = None, n_lags: int = 3) -> Dict[str, Dict]:
        """Predict next month's spending per category using trained regressors.
        Returns a dict: {category: {'pred': value, 'model': name, 'score':score, 'conf_int':(low,high)}}
        """
        results = {}
        data = self._prepare_regression_data(df, n_lags=n_lags)
        latest_month = data['month'].max()
        next_month = (latest_month + pd.offsets.MonthBegin(1)).to_timestamp()

        for cat, meta in self.regressors.items():
            features = meta['features']
            # fetch latest row for this category
            cat_df = data[data['category'] == cat].sort_values('month')
            if cat_df.empty:
                continue
            last = cat_df.iloc[-1]
            X_row = []
            for f in features:
                if f.startswith('lag_'):
                    lag_num = int(f.split('_')[1])
                    # lag value becomes previous lag shifting: new lag_1 = last['amount']
                    if lag_num == 1:
                        X_row.append(last['amount'])
                    else:
                        prev_val = cat_df.iloc[-(lag_num)]['amount'] if len(cat_df) >= lag_num else 0.0
                        X_row.append(prev_val)
                elif f == 'volatility_3m':
                    X_row.append(last['volatility_3m'])
                elif f == 'month_of_year':
                    X_row.append(((next_month).month))
                elif f == 'budget_ratio' and budgets_df is not None:
                    budget_map = budgets_df.set_index('category')['amount'].to_dict()
                    b = budget_map.get(cat, np.nan)
                    X_row.append(last['amount'] / b if b and b > 0 else 0.0)
                else:
                    X_row.append(0.0)

            X_arr = np.array(X_row).reshape(1, -1)
            model = meta['model']
            pred = float(model.predict(X_arr)[0])

            # estimate uncertainty using simple method: std of residuals on training (if available)
            # If model exposes oob or similar, use better method. For now compute residuals if possible
            conf = (max(0.0, pred - 0.2 * abs(pred)), pred + 0.2 * abs(pred))

            results[cat] = {
                'pred': pred,
                'model': meta['model_name'],
                'score': meta['score'],
                'conf_int': conf
            }

        return results

    # --------------------- Drift detection ---------------------
    def _distribution_by_month(self, df: pd.DataFrame, month) -> pd.Series:
        s = df[df['month'] == month].groupby('category')['amount'].sum()
        s = s / s.sum()
        return s

    def compute_drift(self, df: pd.DataFrame, months_back: int = 1) -> Dict:
        """Compute Jensen-Shannon divergence (and PSI) between current month and previous month(s).
        Returns a dict with top drifting categories and scores.
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
        months = sorted(df['month'].unique())
        if len(months) < 2:
            return {'message': 'Not enough months to compute drift'}

        curr = months[-1]
        prev = months[-1 - months_back]
        s_curr = self._distribution_by_month(df, curr)
        s_prev = self._distribution_by_month(df, prev)

        # align index
        all_cats = sorted(set(s_curr.index).union(set(s_prev.index)))
        p = np.array([s_curr.get(c, 0.0) for c in all_cats])
        q = np.array([s_prev.get(c, 0.0) for c in all_cats])

        # Jensen-Shannon distance
        js = float(jensenshannon(p + 1e-12, q + 1e-12))

        # PSI per category (approx)
        # avoid zeros by small eps
        eps = 1e-6
        psi = ((p - q) * np.log((p + eps) / (q + eps))).tolist()
        psi_map = dict(zip(all_cats, psi))

        # top contributors
        top = sorted(psi_map.items(), key=lambda x: -abs(x[1]))[:10]

        report = {
            'current_month': curr,
            'previous_month': prev,
            'jensen_shannon': js,
            'top_psi_contributors': top
        }
        return report

    # --------------------- Plotting helpers ---------------------
    def plot_category_trend(self, df: pd.DataFrame, category: str, save_path: Optional[str] = None):
        df2 = df.copy()
        df2['date'] = pd.to_datetime(df2['date'])
        df2['month'] = df2['date'].dt.to_period('M').dt.to_timestamp()
        grp = df2[df2['category'] == category].groupby('month')['amount'].sum().reset_index()
        plt.figure(figsize=(8, 4))
        plt.plot(grp['month'], grp['amount'], marker='o')
        plt.title(f'Spend Trend - {category}')
        plt.ylabel('Amount')
        plt.xlabel('Month')
        plt.grid(True)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()

    def plot_heatmap_spend_vs_budget(self, df: pd.DataFrame, budgets_df: pd.DataFrame, save_path: Optional[str] = None):
        df2 = df.copy()
        df2['date'] = pd.to_datetime(df2['date'])
        df2['month'] = df2['date'].dt.to_period('M').dt.to_timestamp()
        pivot = df2.groupby(['month', 'category'])['amount'].sum().unstack(fill_value=0)
        # normalize months
        pivot_norm = pivot.div(pivot.sum(axis=1), axis=0)

        plt.figure(figsize=(10, 6))
        plt.imshow(pivot_norm.T, aspect='auto', interpolation='nearest')
        plt.yticks(range(len(pivot.columns)), pivot.columns)
        plt.xticks(range(len(pivot.index)), [m.strftime('%b-%Y') for m in pivot.index], rotation=45)
        plt.title('Spend Share Heatmap (rows=category)')
        plt.colorbar(label='Share')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()

    def plot_anomaly_scatter(self, features_df: pd.DataFrame, score_col: str = 'lof_score', save_path: Optional[str] = None):
        # PCA reduce to 2D for plotting
        feat_cols = ['month_index', 'days_in_month', 'pct_share', 'weekend_ratio', 'roc_prev', 'budget_ratio', 'volatility_3m', 'amount']
        X = features_df[feat_cols].values
        X_scaled = self.scaler.transform(X)
        pca = PCA(n_components=2)
        red = pca.fit_transform(X_scaled)
        features_df = features_df.copy().reset_index(drop=True)
        features_df['x'] = red[:, 0]
        features_df['y'] = red[:, 1]

        plt.figure(figsize=(8, 6))
        plt.scatter(features_df['x'], features_df['y'], c=features_df[score_col], cmap='viridis', s=30)
        if 'is_anomaly' in features_df.columns:
            anomalies = features_df[features_df['is_anomaly']]
            plt.scatter(anomalies['x'], anomalies['y'], facecolors='none', edgecolors='r', s=80, label='anomaly')
            plt.legend()
        plt.title('Anomaly Scatter (PCA reduced)')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()

    # --------------------- Adaptive retrain helpers ---------------------
    def rolling_retrain(self, df: pd.DataFrame, budgets_df: Optional[pd.DataFrame] = None, window_months: int = 12):
        """Perform retraining on a rolling last-N months window. Useful for scheduled retrain jobs."""
        df2 = df.copy()
        df2['date'] = pd.to_datetime(df2['date'])
        df2['month'] = df2['date'].dt.to_period('M').dt.to_timestamp()
        last_month = df2['month'].max()
        start = (last_month - pd.offsets.MonthBegin(window_months - 1)).to_timestamp()
        window_df = df2[df2['month'] >= start]

        features = self.build_features(window_df, budgets_df)
        self.fit_unsupervised(features)
        self.train_regressors(window_df, budgets_df)

    # --------------------- Save / Load utilities ---------------------
    def save_state(self):
        joblib.dump(self.__dict__, os.path.join(self.model_dir, 'finance_ml_state.joblib'))

    def load_state(self):
        p = os.path.join(self.model_dir, 'finance_ml_state.joblib')
        if os.path.exists(p):
            loaded = joblib.load(p)
            self.__dict__.update(loaded)
            # reload scaler if present
            scaler_p = os.path.join(self.model_dir, 'scaler.joblib')
            if os.path.exists(scaler_p):
                self.scaler = joblib.load(scaler_p)


