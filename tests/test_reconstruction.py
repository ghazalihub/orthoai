import unittest
import numpy as np
from orthovision.reconstruction.mesh_generator import MeshGenerator
from orthovision.planning.virtual_reduction import VirtualReduction

class TestReconstruction(unittest.TestCase):

    def test_marching_cubes_logic(self):
        # Create a small sphere-like mask
        mask = np.zeros((20, 20, 20))
        mask[5:15, 5:15, 5:15] = 1

        verts, faces, normals, values = MeshGenerator.generate_mesh_marching_cubes(mask)
        self.assertTrue(len(verts) > 0)
        self.assertTrue(len(faces) > 0)

    def test_transformation(self):
        points = np.array([[1, 0, 0], [0, 1, 0]])
        translation = np.array([1, 1, 1])
        rotation = np.array([0, 0, 90]) # 90 degrees around Z

        transformed = VirtualReduction.apply_transformation(points, translation, rotation)

        # [1, 0, 0] rotated 90 deg around Z is [0, 1, 0]. Then + [1, 1, 1] is [1, 2, 1]
        self.assertAlmostEqual(transformed[0, 0], 1.0)
        self.assertAlmostEqual(transformed[0, 1], 2.0)
        self.assertAlmostEqual(transformed[0, 2], 1.0)

if __name__ == "__main__":
    unittest.main()
