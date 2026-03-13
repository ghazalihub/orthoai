import SimpleITK as sitk
import numpy as np

class MedicalRegistration:
    """
    Handles rigid and non-rigid registration for temporal tracking (healing)
    and multi-modal (CT/X-ray) alignment.
    """

    @staticmethod
    def register_temporal_scans(fixed_image: sitk.Image,
                                moving_image: sitk.Image) -> sitk.Transform:
        """
        Rigid registration to align a follow-up scan to a baseline scan.
        """
        registration_method = sitk.ImageRegistrationMethod()

        # Similarity metric
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.01)

        # Optimizer
        registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
        registration_method.SetOptimizerScalesFromPhysicalShift()

        # Initial Transform
        initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                              moving_image,
                                                              sitk.Euler3DTransform(),
                                                              sitk.CenteredTransformInitializer.GEOMETRY)
        registration_method.SetInitialTransform(initial_transform)

        # Interpolator
        registration_method.SetInterpolator(sitk.sitkLinear)

        final_transform = registration_method.Execute(fixed_image, moving_image)
        return final_transform

    @staticmethod
    def apply_bspline_deformation(fixed_image: sitk.Image,
                                 moving_image: sitk.Image) -> sitk.Image:
        """
        Non-rigid BSpline registration for tracking complex bone remodeling.
        """
        # Complex non-rigid registration logic
        tx = sitk.BSplineTransformInitializer(fixed_image, [8, 8, 8])
        # [Simplified for demo]
        return sitk.Resample(moving_image, fixed_image, tx, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
