import os
import pandas as pd
import numpy as np
from typing import Dict
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import joblib

# --------------------- Regression per category ---------------------
class RegresserML:
    def __init__(self, model_dir: str = 'ml_models'):
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        
        self.regressors: Dict[str, Dict] = {}
        
    def train_regressors(self, df: pd.DataFrame, budgets_df: pd.DataFrame = None, n_lags: int = 3):
        
        categories = df['category'].unique().tolist()

        for cat in categories:
            cat_df = (
                df[df["category"] == cat]
                .sort_values("month")
                .reset_index(drop=True)
            )
            
            # Require enough history
            if len(cat_df) < n_lags + 3:
                continue

            feature_cols = [f'lag_{i}' for i in range(1, n_lags + 1)] + ['volatility_3m', 'month_of_year']
            X = cat_df[feature_cols].copy()
            y = cat_df['amount'].copy()

            # include budget ratio if budgets provided
            if budgets_df is not None:
                budget_map = budgets_df.set_index("category")["monthly_budget"].to_dict()
                b = budget_map.get(cat, np.nan)
                X["budget_ratio"] = (
                    cat_df["lag_1"] / b if b and b > 0 else 0.0
                )
                feature_cols.append("budget_ratio")

            # train/test
            split = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, y_test = y.iloc[:split], y.iloc[split:]

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
                'features': feature_cols,
            }

            # persist
            joblib.dump(self.regressors[cat], os.path.join(self.model_dir, f'regressor_{cat}.joblib'))

    def predict_next_month(self, df: pd.DataFrame, budgets_df: pd.DataFrame = None) -> Dict[str, Dict]:
        """Predict next month's spending per category using trained regressors.
        Returns a dict: {category: {'pred': value, 'model': name, 'score':score, 'conf_int':(low,high)}}
        """
        results = {}
        if df.empty:
            return results
        
        # Ensure we have the right columns
        required_cols = {'month', 'category', 'amount', 'volatility_3m', 'lag_1', 'lag_2', 'lag_3', 'month_of_year'}
        if not required_cols.issubset(df.columns):
            print(f"Warning: Missing columns. Available: {df.columns.tolist()}")
            return results
        
        latest_month = df['month'].max()
        # Calculate next month by adding one month
        next_month = latest_month + pd.DateOffset(months=1)

        budget_map = {}
        if budgets_df is not None:
            budget_map = (
                budgets_df
                .set_index("category")["monthly_budget"]
                .to_dict()
            )

        for cat, meta in self.regressors.items():

            # fetch latest row for this category
            cat_df = df[df['category'] == cat].sort_values('month')

            if cat_df.empty:
                continue

            last = cat_df.iloc[-1]
            X_row = []

            for f in meta["features"]:
                if f.startswith('lag_'):
                    lag_num = int(f.split("_")[1])
                    # Use positional indexing to get lag values
                    if lag_num <= len(cat_df):
                        val = cat_df.iloc[-lag_num][f]
                    else:
                        val = 0.0
                    X_row.append(val)

                elif f == 'volatility_3m':
                    X_row.append(last['volatility_3m'])

                elif f == 'month_of_year':
                    X_row.append(next_month.month)

                elif f == 'budget_ratio':
                    b = budget_map.get(cat, np.nan)
                    # Use lag_1 value for ratio, not last amount
                    lag1_val = cat_df.iloc[-1]['lag_1'] if len(cat_df) > 0 else last['amount']
                    X_row.append(
                        lag1_val / b if b and b > 0 else 0.0
                    )   
                else:
                    X_row.append(0.0)

            # Create DataFrame with feature names to match training
            X_arr = pd.DataFrame([X_row], columns=meta["features"])
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