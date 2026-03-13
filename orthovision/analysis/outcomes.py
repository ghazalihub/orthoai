import numpy as np

class OutcomePredictor:
    """
    Predicts long-term clinical outcomes and surgical complication risks.
    """

    @staticmethod
    def predict_non_union_risk(stability_score: float,
                              bmd_t_score: float,
                              patient_age: int,
                              smoker: bool) -> float:
        """
        Calculates the probability of fracture non-union (healing failure).
        Combines biomechanical, imaging, and clinical factors.
        """
        # Baseline risk
        risk = 0.05

        # Penalties
        if stability_score < 0.5: risk += 0.15
        if bmd_t_score < -2.5: risk += 0.10
        if smoker: risk += 0.08
        if patient_age > 65: risk += 0.05

        return min(risk, 1.0)

    @staticmethod
    def evaluate_implant_longevity(implant_type: str,
                                  activity_level: int,
                                  bone_quality: float) -> dict:
        """
        Predicts years until revision surgery is required.
        """
        # simplified wear-rate model
        base_years = 15.0
        if implant_type == "Titanium-3D": base_years += 5.0

        predicted_life = base_years * (bone_quality / 100.0) / (activity_level + 1)

        return {
            "predicted_life_years": max(predicted_life, 2.0),
            "failure_mode_primary": "Aseptic Loosening"
        }
