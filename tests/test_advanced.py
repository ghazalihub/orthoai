import unittest
import torch
import numpy as np
from orthovision.ai.landmark_detection.gcn_model import AnatomicalGCN
from orthovision.analysis.fracture_geometry import AdvancedFractureGeometry

class TestAdvancedFeatures(unittest.TestCase):

    def test_gcn_refinement_shape(self):
        model = AnatomicalGCN(node_features=3, hidden_channels=16, num_landmarks=5)
        # 5 landmarks, 3 coords each
        x = torch.randn(5, 3)
        # Fully connected graph for 5 nodes (edges: 5*4 = 20)
        edges = []
        for i in range(5):
            for j in range(5):
                if i != j:
                    edges.append([i, j])
        edge_index = torch.tensor(edges).t().contiguous()

        output = model(x, edge_index, None)
        self.assertEqual(output.shape, (5, 3))

    def test_pca_axis_calculation(self):
        # Create points along Z axis
        points = np.zeros((10, 3))
        points[:, 2] = np.linspace(0, 10, 10)
        points += np.random.normal(0, 0.1, (10, 3))

        axis, centroid, _ = AdvancedFractureGeometry.compute_fragment_axis(points)
        # Main axis should be close to [0, 0, 1]
        self.assertAlmostEqual(abs(axis[2]), 1.0, places=1)

if __name__ == "__main__":
    unittest.main()
