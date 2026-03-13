import numpy as np
from scipy.spatial.distance import euclidean

class FractureAnalyzer:
    """
    Analyzes fracture geometry including displacement and angulation.
    """

    @staticmethod
    def measure_displacement(fragment1_centroid: np.ndarray,
                            fragment2_centroid: np.ndarray) -> float:
        """
        Calculates the distance between the centroids of two fragments.
        """
        return euclidean(fragment1_centroid, fragment2_centroid)

    @staticmethod
    def measure_angulation(axis1_start: np.ndarray, axis1_end: np.ndarray,
                           axis2_start: np.ndarray, axis2_end: np.ndarray) -> float:
        """
        Calculates the angulation between the anatomical axes of two fragments.
        """
        v1 = axis1_end - axis1_start
        v2 = axis2_end - axis2_start

        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

        return np.degrees(angle)

    @staticmethod
    def analyze_continuity(bone_mask: np.ndarray) -> int:
        """
        Count the number of connected components to determine comminution.
        """
        from skimage import measure
        labels = measure.label(bone_mask)
        return labels.max()
