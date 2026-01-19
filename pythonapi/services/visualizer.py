import os
import io
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for server environments
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.decomposition import PCA
from datetime import datetime

# Set professional style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

class Visualizer:
    """Generate matplotlib visualizations for financial ML analysis."""
    
    def __init__(self, output_dir: str = None):
        """Initialize visualizer with output directory for saving charts.
        
        Args:
            output_dir: Directory to save generated charts. Defaults to ../reports/charts/
        """
        if output_dir is None:
            # Get directory where this file is located
            current_file = os.path.abspath(__file__)
            pythonapi_dir = os.path.dirname(os.path.dirname(current_file))
            output_dir = os.path.join(pythonapi_dir, 'reports', 'charts')
        
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.chart_paths = {}
    
    def _save_chart(self, name: str, dpi: int = 100) -> str:
        """Save current matplotlib figure and return path.
        
        Args:
            name: Chart name (will be used as filename)
            dpi: Resolution in dots per inch
            
        Returns:
            Absolute path to saved chart
        """
        filepath = os.path.join(self.output_dir, f"{name}.png")
        plt.tight_layout()
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close()
        self.chart_paths[name] = filepath
        return filepath
    
    # ==================== RANGE DATA VISUALIZATIONS ====================
    
    def plot_range_comparison(self, df: pd.DataFrame) -> str:
        """Bar chart comparing spending across categories for selected range.
        
        Args:
            df: DataFrame with 'category' and 'amount' columns
            
        Returns:
            Path to saved chart
        """
        if df.empty:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Aggregate by category
        category_totals = df.groupby('category')['amount'].sum().sort_values(ascending=False)
        
        colors = sns.color_palette("husl", len(category_totals))
        bars = ax.bar(range(len(category_totals)), category_totals.values, color=colors)
        
        ax.set_xticks(range(len(category_totals)))
        ax.set_xticklabels(category_totals.index, rotation=45, ha='right')
        ax.set_ylabel('Total Spending ($)')
        ax.set_title('Spending by Category', fontweight='bold', fontsize=14)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'${height:,.0f}', ha='center', va='bottom', fontsize=9)
        
        return self._save_chart('range_comparison')
    
    def plot_monthly_trends(self, df: pd.DataFrame) -> str:
        """Line chart showing spending trends over time within range.
        
        Args:
            df: DataFrame with 'month' and 'amount' columns
            
        Returns:
            Path to saved chart
        """
        if df.empty or 'month' not in df.columns:
            return None
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Group by month
        monthly_totals = df.groupby('month')['amount'].sum().sort_index()
        
        ax.plot(monthly_totals.index, monthly_totals.values, 
               marker='o', linewidth=2, markersize=8, color='#2E86AB')
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45, ha='right')
        
        ax.set_ylabel('Total Spending ($)')
        ax.set_xlabel('Month')
        ax.set_title('Monthly Spending Trends', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for x, y in zip(monthly_totals.index, monthly_totals.values):
            ax.annotate(f'${y:,.0f}', (x, y), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontsize=8)
        
        return self._save_chart('monthly_trends')
    
    def plot_category_breakdown(self, df: pd.DataFrame) -> str:
        """Pie chart showing category distribution.
        
        Args:
            df: DataFrame with 'category' and 'amount' columns
            
        Returns:
            Path to saved chart
        """
        if df.empty:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        category_totals = df.groupby('category')['amount'].sum().sort_values(ascending=False)
        
        # Combine small categories into "Other"
        threshold = category_totals.sum() * 0.03  # 3% threshold
        main_categories = category_totals[category_totals >= threshold]
        other = category_totals[category_totals < threshold].sum()
        
        if other > 0:
            main_categories['Other'] = other
        
        colors = sns.color_palette("husl", len(main_categories))
        explode = [0.05 if i == 0 else 0 for i in range(len(main_categories))]
        
        wedges, texts, autotexts = ax.pie(main_categories.values, labels=main_categories.index,
                                           autopct='%1.1f%%', colors=colors, explode=explode,
                                           startangle=90, textprops={'fontsize': 10})
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('Spending Distribution by Category', fontweight='bold', fontsize=14)
        
        return self._save_chart('category_breakdown')
    
    def plot_budget_vs_actual(self, budget_metrics: pd.DataFrame) -> str:
        """Comparison chart of budget vs actual spending.
        
        Args:
            budget_metrics: DataFrame with 'category', 'total_spent', 'period_budget' columns
            
        Returns:
            Path to saved chart
        """
        if budget_metrics.empty:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        categories = budget_metrics['category'].tolist()
        actual = budget_metrics['total_spent'].tolist()
        budget = budget_metrics['period_budget'].tolist()
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, actual, width, label='Actual Spending', color='#FF6B6B')
        bars2 = ax.bar(x + width/2, budget, width, label='Budget', color='#4ECDC4')
        
        ax.set_xlabel('Category')
        ax.set_ylabel('Amount ($)')
        ax.set_title('Budget vs Actual Spending', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'${height:,.0f}', ha='center', va='bottom', fontsize=8)
        
        return self._save_chart('budget_vs_actual')
    
    # ==================== ANOMALY DETECTION VISUALIZATIONS ====================
    
    def plot_anomaly_scatter(self, anomaly_df: pd.DataFrame) -> str:
        """2D scatter plot showing anomalies highlighted in red.
        
        Args:
            anomaly_df: DataFrame with anomaly detection results
            
        Returns:
            Path to saved chart
        """
        if anomaly_df.empty:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Use amount vs month_index for scatter
        normal = anomaly_df[~anomaly_df['is_anomaly']]
        anomalies = anomaly_df[anomaly_df['is_anomaly']]
        
        ax.scatter(normal['month_index'], normal['amount'], 
                  c='#4ECDC4', alpha=0.6, s=100, label='Normal', edgecolors='black', linewidth=0.5)
        ax.scatter(anomalies['month_index'], anomalies['amount'], 
                  c='#FF6B6B', alpha=0.8, s=150, label='Anomaly', edgecolors='darkred', linewidth=1.5)
        
        ax.set_xlabel('Month Index')
        ax.set_ylabel('Spending Amount ($)')
        ax.set_title('Anomaly Detection - Spending Patterns', fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return self._save_chart('anomaly_scatter')
    
    def plot_anomaly_timeline(self, anomaly_df: pd.DataFrame) -> str:
        """Timeline view with anomaly markers.
        
        Args:
            anomaly_df: DataFrame with 'month', 'category', 'amount', 'is_anomaly' columns
            
        Returns:
            Path to saved chart
        """
        if anomaly_df.empty or 'month' not in anomaly_df.columns:
            return None
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Group by month and get total with anomaly count
        monthly_data = anomaly_df.groupby('month').agg({
            'amount': 'sum',
            'is_anomaly': 'sum'
        }).sort_index()
        
        # Plot total spending
        ax.plot(monthly_data.index, monthly_data['amount'], 
               marker='o', linewidth=2, markersize=6, color='#2E86AB', label='Total Spending')
        
        # Highlight months with anomalies
        anomaly_months = monthly_data[monthly_data['is_anomaly'] > 0]
        if not anomaly_months.empty:
            ax.scatter(anomaly_months.index, anomaly_months['amount'], 
                      c='#FF6B6B', s=200, zorder=5, label='Months with Anomalies',
                      edgecolors='darkred', linewidth=2, marker='^')
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45, ha='right')
        
        ax.set_ylabel('Total Spending ($)')
        ax.set_xlabel('Month')
        ax.set_title('Anomaly Timeline - Monthly Overview', fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return self._save_chart('anomaly_timeline')
    
    def plot_anomaly_heatmap(self, anomaly_df: pd.DataFrame) -> str:
        """Heatmap showing anomaly intensity by category and month.
        
        Args:
            anomaly_df: DataFrame with 'month', 'category', 'anomaly_votes' columns
            
        Returns:
            Path to saved chart
        """
        if anomaly_df.empty or 'month' not in anomaly_df.columns:
            return None
        
        # Create pivot table
        pivot = anomaly_df.pivot_table(
            values='anomaly_votes', 
            index='category', 
            columns='month',
            aggfunc='max',
            fill_value=0
        )
        
        if pivot.empty:
            return None
        
        fig, ax = plt.subplots(figsize=(14, max(6, len(pivot) * 0.5)))
        
        # Format column labels as month-year
        pivot.columns = [col.strftime('%b %y') if hasattr(col, 'strftime') else str(col) for col in pivot.columns]
        
        sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Anomaly Votes'}, ax=ax, linewidths=0.5)
        
        ax.set_title('Anomaly Heatmap - Category vs Month', fontweight='bold', fontsize=14)
        ax.set_xlabel('Month')
        ax.set_ylabel('Category')
        
        return self._save_chart('anomaly_heatmap')
    
    # ==================== REGRESSION VISUALIZATIONS ====================
    
    def plot_predictions(self, predictions: Dict, historical_df: pd.DataFrame = None) -> str:
        """Bar chart showing predicted vs historical spending by category.
        
        Args:
            predictions: Dict from regression model with category predictions
            historical_df: Optional DataFrame with historical data for comparison
            
        Returns:
            Path to saved chart
        """
        if not predictions:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        categories = list(predictions.keys())
        predicted = [predictions[cat]['pred'] for cat in categories]
        
        # Get historical average if available
        if historical_df is not None and not historical_df.empty:
            historical_avg = historical_df.groupby('category')['amount'].mean()
            historical = [historical_avg.get(cat, 0) for cat in categories]
            
            x = np.arange(len(categories))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, historical, width, label='Historical Avg', color='#95E1D3')
            bars2 = ax.bar(x + width/2, predicted, width, label='Predicted', color='#FF6B6B')
            
            ax.set_xticks(x)
        else:
            bars2 = ax.bar(categories, predicted, color='#FF6B6B', label='Predicted')
        
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.set_ylabel('Spending Amount ($)')
        ax.set_title('Next Month Spending Predictions', fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'${height:,.0f}', ha='center', va='bottom', fontsize=8)
        
        return self._save_chart('predictions')
    
    def plot_prediction_confidence(self, predictions: Dict) -> str:
        """Error bars showing confidence intervals.
        
        Args:
            predictions: Dict from regression model with conf_int
            
        Returns:
            Path to saved chart
        """
        if not predictions:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        categories = list(predictions.keys())
        predicted = [predictions[cat]['pred'] for cat in categories]
        
        # Calculate error bars from confidence intervals
        lower_errors = [predictions[cat]['pred'] - predictions[cat]['conf_int'][0] for cat in categories]
        upper_errors = [predictions[cat]['conf_int'][1] - predictions[cat]['pred'] for cat in categories]
        errors = [lower_errors, upper_errors]
        
        x = np.arange(len(categories))
        ax.errorbar(x, predicted, yerr=errors, fmt='o', markersize=10, 
                   capsize=5, capthick=2, color='#2E86AB', ecolor='#FF6B6B', 
                   elinewidth=2, label='Prediction ± Confidence Interval')
        
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.set_ylabel('Predicted Spending ($)')
        ax.set_title('Prediction Confidence Intervals', fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return self._save_chart('prediction_confidence')
    
    def plot_model_performance(self, predictions: Dict) -> str:
        """Chart showing model scores and accuracy.
        
        Args:
            predictions: Dict from regression model with model scores
            
        Returns:
            Path to saved chart
        """
        if not predictions:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        categories = list(predictions.keys())
        scores = [predictions[cat]['score'] for cat in categories]
        models = [predictions[cat]['model'] for cat in categories]
        
        # Color by model type
        model_colors = {
            'RandomForest': '#4ECDC4',
            'GradientBoosting': '#FF6B6B',
            'XGBoost': '#95E1D3'
        }
        colors = [model_colors.get(m, '#95A5A6') for m in models]
        
        bars = ax.bar(categories, scores, color=colors)
        
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.set_ylabel('R² Score')
        ax.set_ylim([0, 1])
        ax.set_title('Model Performance by Category', fontweight='bold', fontsize=14)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Baseline (0.5)')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add score labels and model names
        for bar, score, model in zip(bars, scores, models):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.2f}\n({model[:2]})', ha='center', va='bottom', fontsize=7)
        
        # Create legend for model types
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=model) 
                          for model, color in model_colors.items() if model in models]
        ax.legend(handles=legend_elements, loc='lower right')
        
        return self._save_chart('model_performance')
    
    # ==================== CLUSTERING VISUALIZATIONS ====================
    
    def plot_cluster_distribution(self, clusters: Dict) -> str:
        """Bar chart showing cluster sizes.
        
        Args:
            clusters: Dict mapping cluster_id to list of descriptions
            
        Returns:
            Path to saved chart
        """
        if not clusters:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        cluster_ids = [str(cid) for cid in clusters.keys()]
        sizes = [len(descs) for descs in clusters.values()]
        
        colors = sns.color_palette("husl", len(cluster_ids))
        bars = ax.bar(cluster_ids, sizes, color=colors)
        
        ax.set_xlabel('Cluster ID')
        ax.set_ylabel('Number of Descriptions')
        ax.set_title('Expense Description Clusters', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add count labels
        for bar, size in zip(bars, sizes):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{size}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        return self._save_chart('cluster_distribution')
    
    def plot_cluster_embeddings(self, embeddings: np.ndarray, labels: List[int]) -> str:
        """2D PCA projection of description embeddings with cluster colors.
        
        Args:
            embeddings: Numpy array of embeddings (n_samples, n_features)
            labels: List of cluster labels
            
        Returns:
            Path to saved chart
        """
        if embeddings is None or len(embeddings) == 0:
            return None
        
        # Reduce to 2D using PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        unique_labels = sorted(set(labels))
        colors = sns.color_palette("husl", len(unique_labels))
        
        for label, color in zip(unique_labels, colors):
            mask = np.array(labels) == label
            label_str = f'Cluster {label}' if label != -1 else 'Outliers'
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                      c=[color], label=label_str, s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax.set_title('Expense Description Embeddings (2D PCA)', fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return self._save_chart('cluster_embeddings')
    
    # ==================== DRIFT DETECTION VISUALIZATIONS ====================
    
    def plot_drift_timeline(self, drift_history: List[Dict]) -> str:
        """Line chart showing drift score over time.
        
        Args:
            drift_history: List of drift results over time
            
        Returns:
            Path to saved chart
        """
        if not drift_history:
            return None
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        months = [d['current_month'] for d in drift_history]
        js_scores = [d['jensen_shannon'] for d in drift_history]
        
        ax.plot(months, js_scores, marker='o', linewidth=2, markersize=8, color='#2E86AB')
        
        # Add threshold lines
        ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='Low drift threshold')
        ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='High drift threshold')
        
        ax.set_ylabel('Jensen-Shannon Distance')
        ax.set_xlabel('Month')
        ax.set_title('Spending Pattern Drift Over Time', fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        
        return self._save_chart('drift_timeline')
    
    def plot_drift_contributors(self, drift_result: Dict) -> str:
        """Bar chart of top PSI contributors.
        
        Args:
            drift_result: Drift analysis result with top_psi_contributors
            
        Returns:
            Path to saved chart
        """
        if not drift_result or 'top_psi_contributors' not in drift_result:
            return None
        
        contributors = drift_result['top_psi_contributors'][:10]  # Top 10
        
        if not contributors:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        categories = [c[0] for c in contributors]
        psi_values = [abs(c[1]) for c in contributors]
        colors = ['#FF6B6B' if c[1] > 0 else '#4ECDC4' for c in contributors]
        
        bars = ax.barh(categories, psi_values, color=colors)
        
        ax.set_xlabel('Absolute PSI Value')
        ax.set_ylabel('Category')
        ax.set_title('Top Contributors to Spending Drift', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, val in zip(bars, psi_values):
            ax.text(val, bar.get_y() + bar.get_height()/2.,
                   f'{val:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FF6B6B', label='Increased spending'),
            Patch(facecolor='#4ECDC4', label='Decreased spending')
        ]
        ax.legend(handles=legend_elements)
        
        return self._save_chart('drift_contributors')
    
    def plot_category_distribution_shift(self, df: pd.DataFrame, drift_result: Dict) -> str:
        """Side-by-side comparison of category distributions.
        
        Args:
            df: DataFrame with monthly category data
            drift_result: Drift result with current_month and previous_month
            
        Returns:
            Path to saved chart
        """
        if df.empty or not drift_result:
            return None
        
        curr_month = pd.to_datetime(drift_result['current_month'])
        prev_month = pd.to_datetime(drift_result['previous_month'])
        
        curr_data = df[df['month'] == curr_month].groupby('category')['amount'].sum()
        prev_data = df[df['month'] == prev_month].groupby('category')['amount'].sum()
        
        if curr_data.empty or prev_data.empty:
            return None
        
        # Normalize to percentages
        curr_pct = (curr_data / curr_data.sum() * 100).sort_values(ascending=True)
        prev_pct = (prev_data / prev_data.sum() * 100).reindex(curr_pct.index, fill_value=0)
        
        fig, ax = plt.subplots(figsize=(12, max(6, len(curr_pct) * 0.4)))
        
        y = np.arange(len(curr_pct))
        height = 0.35
        
        ax.barh(y - height/2, prev_pct.values, height, label=f"Previous ({prev_month.strftime('%b %Y')})", color='#95E1D3')
        ax.barh(y + height/2, curr_pct.values, height, label=f"Current ({curr_month.strftime('%b %Y')})", color='#FF6B6B')
        
        ax.set_yticks(y)
        ax.set_yticklabels(curr_pct.index)
        ax.set_xlabel('Percentage of Total Spending (%)')
        ax.set_title('Category Distribution Shift', fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        return self._save_chart('category_distribution_shift')
    
    def get_all_chart_paths(self) -> Dict[str, str]:
        """Get dictionary of all generated chart paths.
        
        Returns:
            Dict mapping chart names to file paths
        """
        return self.chart_paths.copy()
    
    def clear_charts(self):
        """Delete all generated chart files and clear paths."""
        for filepath in self.chart_paths.values():
            if os.path.exists(filepath):
                os.remove(filepath)
        self.chart_paths.clear()
