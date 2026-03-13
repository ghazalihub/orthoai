import unittest
import torch
from orthovision.ai.segmentation.model import get_bone_segmentation_model
from orthovision.ai.landmark_detection.model import get_landmark_detection_model

class TestAIModels(unittest.TestCase):

    def test_segmentation_model_output_shape(self):
        model = get_bone_segmentation_model(in_channels=1, out_channels=2)
        dummy_input = torch.randn(1, 1, 32, 32, 32)
        output = model(dummy_input)
        self.assertEqual(output.shape, (1, 2, 32, 32, 32))

    def test_landmark_model_output_shape(self):
        model = get_landmark_detection_model(in_channels=1, out_landmarks=5)
        dummy_input = torch.randn(1, 1, 32, 32, 32)
        output = model(dummy_input)
        self.assertEqual(output.shape, (1, 5, 32, 32, 32))

if __name__ == "__main__":
    unittest.main()
