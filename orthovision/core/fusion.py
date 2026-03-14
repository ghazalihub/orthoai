import SimpleITK as sitk
import numpy as np

class MultiModalFusion:
    """
    Registers and fuses CT bone models with MRI soft tissue masks.
    Enables planning of ligaments reconstruction (e.g., ACL) in the context of bone geometry.
    """

    @staticmethod
    def fuse_ct_mri(ct_image: sitk.Image,
                   mri_image: sitk.Image) -> sitk.Transform:
        """
        Multimodal registration using Mutual Information.
        """
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
        registration_method.SetInitialTransform(sitk.CenteredTransformInitializer(ct_image, mri_image, sitk.Euler3DTransform()))
        registration_method.SetInterpolator(sitk.sitkLinear)

        return registration_method.Execute(ct_image, mri_image)

    @staticmethod
    def overlay_masks(bone_mask: np.ndarray,
                     soft_tissue_mask: np.ndarray) -> np.ndarray:
        """
        Combines CT and MRI masks into a single hybrid model.
        """
        # Values: 1-Bone, 2-ACL, 3-Meniscus, etc.
        fused = bone_mask.copy()
        fused[soft_tissue_mask > 0] = soft_tissue_mask[soft_tissue_mask > 0] + 1
        return fused
