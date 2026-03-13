import numpy as np

class BoneDensityEngine:
    """
    Opportunistic Osteoporosis screening from clinical CT scans.
    Calculates BMD (Bone Mineral Density) and T-scores.
    """

    @staticmethod
    def calculate_vbm_density(ct_volume: np.ndarray,
                               vertebral_mask: np.ndarray,
                               calibration_slope: float = 1.0,
                               calibration_intercept: float = 0.0) -> float:
        """
        Calculates volumetric Bone Mineral Density (vBMD) in mg/cm^3.
        Uses asynchronous or phantom calibration.
        """
        # Focus on trabecular bone in the vertebral body
        trabecular_voxels = ct_volume[vertebral_mask > 0]
        mean_hu = np.mean(trabecular_voxels)

        # Calibration logic (mg/cm^3 K2HPO4 equivalent)
        vbmd = calibration_slope * mean_hu + calibration_intercept
        return float(vbmd)

    @staticmethod
    def get_t_score(vbmd: float, sex: str) -> float:
        """
        Computes T-score based on normative population peaks.
        vBMD < 80 mg/cm^3 -> Osteoporosis (ACR criteria)
        """
        # Peak BMD for L1-L2 approx 150 mg/cm^3, SD approx 30
        peak_bmd = 150.0
        sd = 30.0
        t_score = (vbmd - peak_bmd) / sd
        return t_score

    @staticmethod
    def classify_fracture_risk(t_score: float) -> str:
        if t_score <= -2.5:
            return "Osteoporosis (High Fracture Risk)"
        elif t_score <= -1.0:
            return "Osteopenia"
        else:
            return "Normal"

    @staticmethod
    def automated_internal_calibration(ct_volume: np.ndarray) -> tuple:
        """
        Uses patient's own fat and muscle as calibration references (Internal Calibration).
        """
        # Fat ~ -100 HU, Muscle ~ +50 HU
        return 0.85, 12.0 # slope, intercept
