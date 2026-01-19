import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

class DriftDetectionML:
    """Compute distribution drift between time periods using Jensen-Shannon divergence."""
    
    @staticmethod
    def _distribution_by_month(df: pd.DataFrame, month) -> pd.Series:
        """Get spending distribution by category for a given month."""
        s = df[df['month'] == month].groupby('category')['amount'].sum()
        s = s / s.sum() if s.sum() > 0 else s
        return s

    @staticmethod
    def compute_drift(df: pd.DataFrame, months_back: int = 1) -> dict:
        """Compute Jensen-Shannon divergence between current month and previous month(s).
        
        Args:
            df: DataFrame with columns ['month', 'category', 'amount']
            months_back: How many months back to compare (default: 1 for previous month)
            
        Returns:
            Dict with current_month, previous_month, jensen_shannon score, and top PSI contributors
        """
        df = df.copy()
        months = sorted(df['month'].unique())
        
        if len(months) < 2:
            return {'message': 'Not enough months to compute drift'}

        curr = months[-1]
        prev_idx = len(months) - 1 - months_back
        prev = months[prev_idx] if prev_idx >= 0 else months[0]
        
        s_curr = DriftDetectionML._distribution_by_month(df, curr)
        s_prev = DriftDetectionML._distribution_by_month(df, prev)

        # Align categories
        all_cats = sorted(set(s_curr.index).union(set(s_prev.index)))
        p = np.array([s_curr.get(c, 0.0) for c in all_cats])
        q = np.array([s_prev.get(c, 0.0) for c in all_cats])

        # Jensen-Shannon distance
        js = float(jensenshannon(p + 1e-12, q + 1e-12))

        # PSI (Population Stability Index) per category
        eps = 1e-6
        psi = ((p - q) * np.log((p + eps) / (q + eps))).tolist()
        psi_map = dict(zip(all_cats, psi))

        # Top contributors to drift
        top = sorted(psi_map.items(), key=lambda x: -abs(x[1]))[:10]

        report = {
            'current_month': str(curr),
            'previous_month': str(prev),
            'jensen_shannon': js,
            'top_psi_contributors': top
        }
        return report

    @staticmethod
    def compute_drift_history(df: pd.DataFrame, max_periods: int = 6) -> list:
        """Compute drift history over multiple consecutive months.
        
        Args:
            df: DataFrame with columns ['month', 'category', 'amount']
            max_periods: Maximum number of month-to-month comparisons
            
        Returns:
            List of drift reports for consecutive month pairs
        """
        df = df.copy()
        months = sorted(df['month'].unique())
        
        if len(months) < 2:
            return []
        
        drift_history = []
        
        # Compute drift for each consecutive month pair
        num_comparisons = min(max_periods, len(months) - 1)
        start_idx = len(months) - num_comparisons - 1
        
        for i in range(start_idx, len(months) - 1):
            curr = months[i + 1]
            prev = months[i]
            
            s_curr = DriftDetectionML._distribution_by_month(df, curr)
            s_prev = DriftDetectionML._distribution_by_month(df, prev)
            
            # Align categories
            all_cats = sorted(set(s_curr.index).union(set(s_prev.index)))
            p = np.array([s_curr.get(c, 0.0) for c in all_cats])
            q = np.array([s_prev.get(c, 0.0) for c in all_cats])
            
            # Jensen-Shannon distance
            js = float(jensenshannon(p + 1e-12, q + 1e-12))
            
            drift_history.append({
                'current_month': str(curr),
                'previous_month': str(prev),
                'jensen_shannon': js
            })
        
        return drift_history
