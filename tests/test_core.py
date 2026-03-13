import unittest
import SimpleITK as sitk
import numpy as np
from orthovision.core.dicom_engine import DICOMEngine
from orthovision.core.preprocessing import PreprocessingPipeline

class TestCore(unittest.TestCase):

    def test_preprocessing_isotropic_resampling(self):
        # Create a dummy image
        image = sitk.GetImageFromArray(np.zeros((64, 64, 64), dtype=np.float32))
        image.SetSpacing([1.0, 1.0, 2.0])

        pipeline = PreprocessingPipeline()
        resampled_image = pipeline.resample_to_isotropic(image, new_spacing=1.0)

        new_spacing = resampled_image.GetSpacing()
        new_size = resampled_image.GetSize()

        self.assertEqual(new_spacing, (1.0, 1.0, 1.0))
        self.assertEqual(new_size, (64, 64, 128))

    def test_hu_normalization(self):
        # Create a dummy image with known HU values
        arr = np.array([-1000.0, 0.0, 1000.0, 2000.0, 3000.0], dtype=np.float32).reshape((1, 1, 5))
        image = sitk.GetImageFromArray(arr)

        pipeline = PreprocessingPipeline()
        normalized_image = pipeline.normalize_hu(image, min_hu=-1000, max_hu=2000)

        normalized_arr = sitk.GetArrayFromImage(normalized_image)

        self.assertAlmostEqual(normalized_arr[0, 0, 0], 0.0, places=5)
        self.assertAlmostEqual(normalized_arr[0, 0, 3], 1.0, places=5)
        self.assertAlmostEqual(normalized_arr[0, 0, 4], 1.0, places=5) # Clamped

if __name__ == "__main__":
    unittest.main()
