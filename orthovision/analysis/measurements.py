import numpy as np
from typing import Tuple, List

class OrthopedicMeasurements:
    """
    Calculates orthopedic parameters from landmark coordinates.
    """

    @staticmethod
    def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Calculates the angle at p2 formed by p1-p2 and p2-p3.
        """
        v1 = p1 - p2
        v2 = p3 - p2

        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

        return np.degrees(angle)

    @staticmethod
    def calculate_cobb_angle(line1_start: np.ndarray, line1_end: np.ndarray,
                             line2_start: np.ndarray, line2_end: np.ndarray) -> float:
        """
        Calculates the Cobb angle between two lines (e.g., endplates of vertebrae).
        """
        v1 = line1_end - line1_start
        v2 = line2_end - line2_start

        # Angle between vectors
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

        return np.degrees(angle)

    @staticmethod
    def calculate_neck_shaft_angle(femur_head_center: np.ndarray,
                                  neck_center: np.ndarray,
                                  shaft_mid_point: np.ndarray) -> float:
        """
        Calculates the femoral neck-shaft angle.
        """
        return OrthopedicMeasurements.calculate_angle(femur_head_center, neck_center, shaft_mid_point)

    @staticmethod
    def calculate_mechanical_axis(hip_center: np.ndarray,
                                 knee_center: np.ndarray,
                                 ankle_center: np.ndarray) -> float:
        """
        Calculates the HKA (Hip-Knee-Ankle) angle.
        """
        return OrthopedicMeasurements.calculate_angle(hip_center, knee_center, ankle_center)
