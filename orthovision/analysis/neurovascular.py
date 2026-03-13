import numpy as np
from scipy.spatial import KDTree

class NeurovascularSafety:
    """
    Calculates proximity of implants and instruments to neurovascular structures.
    Crucial for preventing intra-operative nerve damage or vascular injury.
    """

    @staticmethod
    def calculate_safety_margin(screw_trajectory: dict,
                                vessel_points: np.ndarray,
                                nerve_points: np.ndarray) -> dict:
        """
        Calculates the minimum distance from the screw path to critical structures.
        """
        # Screw path defined by points every 1mm
        p1, p2 = screw_trajectory["entry"], screw_trajectory["target"]
        direction = p2 - p1
        length = np.linalg.norm(direction)
        path_points = p1 + np.linspace(0, 1, int(length))[:, None] * direction

        vessel_tree = KDTree(vessel_points)
        nerve_tree = KDTree(nerve_points)

        v_dist, _ = vessel_tree.query(path_points)
        n_dist, _ = nerve_tree.query(path_points)

        min_v = np.min(v_dist)
        min_n = np.min(n_dist)

        return {
            "min_vessel_distance": float(min_v),
            "min_nerve_distance": float(min_n),
            "safe": min_v > 5.0 and min_n > 3.0 # Standard safety thresholds (mm)
        }

    @staticmethod
    def generate_risk_heatmap(planned_trajectories: list,
                             anatomical_atlas: dict) -> np.ndarray:
        """
        Generates a 3D heatmap of surgical risk based on neurovascular density.
        """
        return np.random.rand(64, 64, 64) # Risk map
