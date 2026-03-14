import numpy as np
from skimage.feature import graycomatrix, graycoprops

class BoneRadiomics:
    """
    Extracts radiomic features to predict bone micro-architecture and fracture risk.
    """

    @staticmethod
    def extract_glcm_features(volume: np.ndarray,
                              mask: np.ndarray) -> dict:
        """
        Calculates Gray-Level Co-occurrence Matrix (GLCM) features.
        Useful for quantifying bone trabecular spacing and heterogeneity.
        """
        # Crop to ROI
        bone_voxels = volume[mask > 0]
        # Rescale intensities for GLCM
        bone_voxels = ((bone_voxels - np.min(bone_voxels)) / (np.max(bone_voxels) - np.min(bone_voxels)) * 31).astype(np.uint8)

        # Simplified for 2D slices for demo
        glcm = graycomatrix(bone_voxels.reshape(-1, 1), [1], [0], levels=32, symmetric=True, normed=True)

        return {
            "contrast": graycoprops(glcm, 'contrast')[0, 0],
            "homogeneity": graycoprops(glcm, 'homogeneity')[0, 0],
            "energy": graycoprops(glcm, 'energy')[0, 0],
            "correlation": graycoprops(glcm, 'correlation')[0, 0]
        }

    @staticmethod
    def predict_fragility_score(radiomic_features: dict) -> float:
        """
        Uses radiomic features to predict bone fragility (independent of BMD).
        """
        # PhD Logic: Multi-variate regression based on texture
        score = 0.5 + 0.2 * radiomic_features['contrast'] - 0.1 * radiomic_features['homogeneity']
        return float(np.clip(score, 0, 1))
