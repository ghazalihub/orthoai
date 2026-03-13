import SimpleITK as sitk
import numpy as np

class PreprocessingPipeline:
    """
    Pipeline for normalizing and resampling medical images.
    """

    @staticmethod
    def resample_to_isotropic(image: sitk.Image, new_spacing: float = 1.0) -> sitk.Image:
        """
        Resample a SimpleITK image to a uniform isotropic spacing.
        """
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()

        new_size = [
            int(round(original_size[i] * (original_spacing[i] / new_spacing)))
            for i in range(len(original_size))
        ]

        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(new_size)
        resampler.SetOutputSpacing([new_spacing] * len(original_size))
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetInterpolator(sitk.sitkLinear)

        return resampler.Execute(image)

    @staticmethod
    def normalize_hu(image: sitk.Image, min_hu: float = -1000.0, max_hu: float = 2000.0) -> sitk.Image:
        """
        Normalize Hounsfield Units (HU) of a CT image to a fixed range.
        Typically for orthopedic CT, bones range from 200 to 2000+ HU.
        """
        clamp_filter = sitk.ClampImageFilter()
        clamp_filter.SetLowerBound(min_hu)
        clamp_filter.SetUpperBound(max_hu)
        image = clamp_filter.Execute(image)

        # Scale to [0, 1] range
        rescale_filter = sitk.RescaleIntensityImageFilter()
        rescale_filter.SetOutputMinimum(0.0)
        rescale_filter.SetOutputMaximum(1.0)
        image = rescale_filter.Execute(image)

        return image

    @staticmethod
    def z_score_normalization(image: sitk.Image) -> sitk.Image:
        """
        Apply Z-score normalization to an image.
        """
        img_array = sitk.GetArrayFromImage(image)
        mean = np.mean(img_array)
        std = np.std(img_array)

        normalized_array = (img_array - mean) / (std + 1e-8)

        normalized_image = sitk.GetImageFromArray(normalized_array)
        normalized_image.CopyInformation(image)

        return normalized_image
