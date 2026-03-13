import numpy as np
import pandas as pd

class PopulationMorphometrics:
    """
    Analyzes anatomical variation across large populations.
    Used for designing universal implants or identifying population-specific risks.
    """

    def __init__(self, data_frame: pd.DataFrame):
        self.df = data_frame # Contains measurements for thousands of patients

    def compute_bone_atlas_statistics(self) -> dict:
        """
        Calculates normative means and standard deviations for anatomical parameters.
        """
        stats = {
            "neck_shaft_angle": {
                "mean": self.df["nsa"].mean(),
                "std": self.df["nsa"].std(),
                "p95": np.percentile(self.df["nsa"], 95)
            },
            "canal_flare_index": {
                "mean": self.df["cfi"].mean()
            }
        }
        return stats

    def cluster_anatomical_phenotypes(self, n_clusters: int = 3):
        """
        Uses K-means to identify common anatomical subtypes (e.g., Coxa Vara vs Valga).
        """
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters)
        features = self.df[["nsa", "femoral_version", "head_diameter"]]
        self.df["phenotype"] = kmeans.fit_predict(features)
        return self.df
