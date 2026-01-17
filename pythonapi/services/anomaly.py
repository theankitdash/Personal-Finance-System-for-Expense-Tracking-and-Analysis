from typing import List, Optional
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import torch
import torch.nn as nn
from torch.optim import Adam as TorchAdam
from torch.utils.data import TensorDataset, DataLoader

# --------------------- Unsupervised anomaly models ---------------------
class AnomalyML:
    def __init__(self, model_dir: str = None):
        if model_dir is None:
            # Get directory where this file is located
            current_file = os.path.abspath(__file__)  # /path/to/pythonapi/services/anomaly.py
            pythonapi_dir = os.path.dirname(os.path.dirname(current_file))  # /path/to/pythonapi/
            model_dir = os.path.join(pythonapi_dir, 'ml_models')
        
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.scaler = StandardScaler()

        # Placeholders for models
        self.lof = None
        self.ocsvm = None
        self.autoencoder = None
        self.autoencoder_threshold = None
        self.autoencoder_device = None

    def fit_unsupervised(self, features_df: pd.DataFrame, feature_columns: Optional[List[str]] = None):
        """Fit LOF, One-Class SVM, and Autoencoder on the passed features dataframe.
        Saves models to disk under model_dir.
        """
        if feature_columns is None:
            feature_columns = ['month_index', 'days_in_month', 'pct_share', 'weekend_ratio', 'roc_prev', 'budget_ratio', 'volatility_3m', 'amount']

        X = features_df[feature_columns].values.astype(float)
        X_scaled = self.scaler.fit_transform(X)

        # LOF
        n_samples = X_scaled.shape[0]
        # Use simple heuristic: min(20, n_samples - 1) but ensuring at least 1 neighbor if possible
        n_neighbors = min(20, max(1, n_samples - 1))
        
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.05, novelty=True)
        lof.fit(X_scaled)
        self.lof = lof
        joblib.dump(lof, os.path.join(self.model_dir, 'lof.joblib'))

        # One-Class SVM
        oc = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
        oc.fit(X_scaled)
        self.ocsvm = oc
        joblib.dump(oc, os.path.join(self.model_dir, 'ocsvm.joblib'))

        # Autoencoder (optional)
        
        dim = X_scaled.shape[1]
        encoding_dim = max(4, dim // 2)
        
        # Define PyTorch autoencoder
        class Autoencoder(nn.Module):
            def __init__(self, input_dim, encoding_dim):
                super(Autoencoder, self).__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, encoding_dim * 2),
                    nn.ReLU(),
                    nn.Linear(encoding_dim * 2, encoding_dim),
                    nn.ReLU()
                )
                self.decoder = nn.Sequential(
                    nn.Linear(encoding_dim, encoding_dim * 2),
                    nn.ReLU(),
                    nn.Linear(encoding_dim * 2, input_dim)
                )
            
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded
            
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ae = Autoencoder(dim, encoding_dim).to(device)
        optimizer = TorchAdam(ae.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Convert data to tensor
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Train with early stopping
        best_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        for epoch in range(200):
            epoch_loss = 0.0
            for batch in dataloader:
                X_batch = batch[0]
                optimizer.zero_grad()
                recon = ae(X_batch)
                loss = criterion(recon, X_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            epoch_loss /= len(dataloader)
            
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        self.autoencoder = ae
        
        # compute reconstruction errors and threshold (mean + 3*std)
        ae.eval()
        with torch.no_grad():
            recon = ae(X_tensor)
            mse = torch.mean((recon - X_tensor) ** 2, dim=1).cpu().numpy()
        thresh = np.mean(mse) + 3 * np.std(mse)
        self.autoencoder_threshold = float(thresh)
        self.autoencoder_device = device
        
        # Save
        torch.save(ae.state_dict(), os.path.join(self.model_dir, 'autoencoder_pytorch.pth'))
        

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
                lof_scores = np.zeros(X.shape[0])
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
            self.autoencoder.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled).to(self.autoencoder_device)
                recon = self.autoencoder(X_tensor)
                mse = torch.mean((recon - X_tensor) ** 2, dim=1).cpu().numpy()
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
