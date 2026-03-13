import numpy as np

class HealingScoreCalculator:
    """
    Implements mRUST (Modified Radiographic Union Scale for Tibial fractures).
    Analyzes cortical bridging in 4 views.
    """

    @staticmethod
    def calculate_mrust(cortical_scores: list) -> int:
        """
        cortical_scores: List of 4 scores (one for each cortex: anterior, posterior, medial, lateral).
        Score per cortex:
        1: Fracture line visible, no callus
        2: Visible callus, fracture line visible
        3: Visible callus, fracture line bridging
        4: Remodeled bone, no fracture line

        Total score: 4 to 16
        """
        if len(cortical_scores) != 4:
            raise ValueError("mRUST requires 4 cortical scores.")
        return sum(cortical_scores)

    @staticmethod
    def automated_callus_analysis(pre_img: np.ndarray, post_img: np.ndarray) -> float:
        """
        Calculates the Callus Volume Index (CVI) by comparing temporal scans.
        """
        diff = post_img - pre_img
        callus_voxels = np.sum(diff > 100) # Threshold for new bone/callus
        return float(callus_voxels)
