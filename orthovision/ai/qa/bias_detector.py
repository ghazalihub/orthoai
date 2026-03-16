import numpy as np

class OrthoBiasAuditor:
    """
    Evaluates orthopedic AI performance across demographic groups to ensure fairness.
    Checks for performance gaps in age, sex, and ethnicity.
    """

    @staticmethod
    def calculate_disparate_impact(performance_metrics: dict) -> dict:
        """
        Calculates the ratio of performance between different groups.
        Goal: Impact Ratio > 0.8 (Four-Fifths Rule).
        """
        # metrics: {"male": 0.92, "female": 0.88, "pediatric": 0.85}
        base_performance = max(performance_metrics.values())
        impact_ratios = {k: v / base_performance for k, v in performance_metrics.items()}

        return impact_ratios

    @staticmethod
    def audit_segmentation_bias(dice_scores_by_age: dict) -> bool:
        """
        Checks if bone segmentation accuracy drops significantly in elderly patients
        (due to osteoporosis/low contrast).
        """
        elderly_avg = dice_scores_by_age.get("70+", 0.0)
        adult_avg = dice_scores_by_age.get("20-60", 1.0)

        return (elderly_avg / adult_avg) > 0.9
