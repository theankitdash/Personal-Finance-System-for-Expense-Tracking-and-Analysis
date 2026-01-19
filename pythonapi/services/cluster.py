from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer
from typing import List
import pandas as pd
import numpy as np

CONCEPT_MAP = {
    "lunch": "meal",
    "dinner": "meal",
    "breakfast": "meal",
    "snack": "meal",

    "milk": "dairy",
    "curd": "dairy",
    "yogurt": "dairy",

    "fastag": "toll",

    "bike service": "vehicle_service",
    "car service": "vehicle_service",

    "laptop": "electronics",

    "gift": "gift",
    "gift wrap": "gift"
}

def normalize_description(text: str) -> str:
    text = text.lower().strip()

    for key, concept in CONCEPT_MAP.items():
        if key in text:
            return concept

    return text

# --------------------- Description embeddings & clustering ---------------------    
class ClusterML: 
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.description_embeddings = None
        self.cluster_labels = None
        self.cluster_model = None
        self.sentence_model = SentenceTransformer(model_name)  

    def fit_description_embeddings(self, descriptions: List[str]) -> pd.DataFrame:
        
        texts = [str(d) for d in descriptions]
        embeddings = self.sentence_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )
        df = pd.DataFrame(embeddings, index=texts)
        self.description_embeddings = df
        return df

    @staticmethod
    def find_best_k(embeddings, k_min=2, k_max=8):
       
        best_k = k_min
        best_score = -1

        max_k = min(k_max, len(embeddings) - 1)

        for k in range(k_min, max_k + 1):
            km = KMeans(
                n_clusters=k,
                n_init=20,
                max_iter=500,
                tol=1e-5,
                random_state=42
            )
            labels = km.fit_predict(embeddings)

            # silhouette needs at least 2 clusters and no empty clusters
            if len(set(labels)) > 1:
                score = silhouette_score(embeddings, labels)
                if score > best_score:
                    best_score = score
                    best_k = k

        return best_k

    def cluster_descriptions_kmeans(self, k_min=2, k_max=8, outlier_percentile=90):
        if self.description_embeddings is None:
            raise ValueError('Call fit_description_embeddings() first')

        X = self.description_embeddings.values

        # 1. choose k intelligently
        best_k = self.find_best_k(X, k_min=k_min, k_max=k_max)

        # 2. final clustering with best k
        km = KMeans(
            n_clusters=best_k,
            n_init=50,
            max_iter=500,
            tol=1e-5,
            random_state=42
        )
        
        labels = km.fit_predict(X)

        # 3. distance to centroids (confidence measure)
        distances = cdist(X, km.cluster_centers_, metric="euclidean")
        min_distances = distances[np.arange(len(distances)), labels]

        # 4. detect weak assignments (outliers)
        final_labels = []
        for i, (label, dist) in enumerate(zip(labels, min_distances)):
            cluster_dists = min_distances[labels == label]
            cluster_threshold = np.percentile(cluster_dists, outlier_percentile)
            final_labels.append(label if dist <= cluster_threshold else -1)

        # 5. store everything useful
        self.cluster_model = km
        self.n_clusters = best_k
        self.cluster_centers = km.cluster_centers_
        self.cluster_distances = dict(zip(self.description_embeddings.index.tolist(), min_distances.tolist()))

        self.cluster_labels = dict(zip(self.description_embeddings.index.tolist(), final_labels))

        return self.cluster_labels

    def merge_semantic_descriptions(self, df: pd.DataFrame) -> pd.DataFrame:

        if self.cluster_labels is None:
            raise ValueError("Run embedding + clustering first")
        
        df2 = df.copy()
        df2['description_cluster'] = df2['description'].map(lambda d: f'cluster_{self.cluster_labels.get(str(d), -1)}')
        
        return df2

    def get_embeddings_and_labels(self):
        """Get embeddings and labels for visualization.
        
        Returns:
            Tuple of (embeddings_array, labels_list) or (None, None) if not fitted
        """
        if self.description_embeddings is None or self.cluster_labels is None:
            return None, None
        
        embeddings = self.description_embeddings.values
        # Get labels in same order as embeddings
        labels = [self.cluster_labels.get(desc, -1) for desc in self.description_embeddings.index]
        
        return embeddings, labels
