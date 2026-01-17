import numpy as np
import pandas as pd

# Data preparation for ML models
class DataPrepML:
    @staticmethod
    def build_base_transactions(range_data, all_data):

        df_range = pd.DataFrame(range_data) 
        df_all = pd.DataFrame(all_data)

        for df in [df_range, df_all]:
            if df.empty:
                # Ensure columns exist even if empty to prevent downstream errors
                for col in ['date', 'amount', 'category', 'description']:
                    if col not in df.columns:
                        df[col] = []
                continue

            df['date'] = pd.to_datetime(df['date'])
            df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0)

        return df_range, df_all    

    @staticmethod 
    def build_monthly_category(df_range):

        # Group by month & category
        grp = (
            df_range
            .groupby(['month', 'category'])['amount']
            .sum()
            .reset_index()
        )
        # Monthly totals
        month_total = (
            grp.groupby('month')['amount']
            .sum()
            .rename('month_total')
            .reset_index()
        )

        grp = grp.merge(month_total, on='month', how='left')

        grp['pct_share'] = (
            grp['amount'] /
            grp['month_total'].replace(0, np.nan)
        ).fillna(0.0)

        return grp
    
    @staticmethod
    def get_timeseries_features(grp, n_lags=3):

        df = grp.copy()
        
        df['month_index'] = (
            (df['month'].dt.year - df['month'].min().year) * 12 +
            (df['month'].dt.month - df['month'].min().month)
        ).astype(int)

        df['month_of_year'] = df['month'].dt.month
        df['days_in_month'] = df['month'].dt.daysinmonth
        df['avg_daily_spend'] = df['amount'] / df['days_in_month']

        for lag in range(1, n_lags + 1):
            df[f'lag_{lag}'] = (
                df.groupby('category')['amount']
                .shift(lag)
                .fillna(0.0)
            )
        df['rolling_mean_3m'] = (
            df.groupby('category')['amount']
            .rolling(3, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        return df
    
    @staticmethod
    def build_monthly_behavioral_features(df_tx: pd.DataFrame) -> pd.DataFrame:

        tx = df_tx.copy()
        tx['is_weekend'] = tx['date'].dt.dayofweek >= 5

        monthly_behavior = (
            tx.groupby(['month', 'category'])['is_weekend']
            .mean()
            .rename('weekend_ratio')
            .reset_index()
        )

        return monthly_behavior


    @staticmethod
    def get_anomaly_features(grp):
        grp = grp.copy()
        grp['roc_prev'] = grp.groupby('category')['amount'].pct_change().fillna(0.0)
        
        grp['volatility_3m'] = (
            grp.groupby('category')['amount']
            .rolling(3, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
            .fillna(0.0)
        )

        return grp[['roc_prev', 'volatility_3m']]
        
    @staticmethod
    def compute_budget_metrics(features, budget_data, num_months):
        df_budget = pd.DataFrame(budget_data)
        df_budget['amount'] = pd.to_numeric(df_budget['amount'], errors='coerce').fillna(0.0)

        budgets = df_budget.set_index('category')['amount']

        category_period = (
            features.groupby('category')['amount']
            .agg(total_spent='sum', avg_monthly_spent='mean')
            .reset_index()
        )

        category_period['monthly_budget'] = category_period['category'].map(budgets)
        category_period['period_budget'] = category_period['monthly_budget'] * num_months

        category_period['budget_ratio'] = (
            category_period['total_spent'] /
            category_period['period_budget'].replace(0, np.nan)
        ).fillna(0.0)

        return category_period
