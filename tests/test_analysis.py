import unittest
import numpy as np
from orthovision.analysis.measurements import OrthopedicMeasurements
from orthovision.analysis.fracture_analyzer import FractureAnalyzer

class TestAnalysis(unittest.TestCase):

    def test_angle_calculation(self):
        p1 = np.array([1, 0, 0])
        p2 = np.array([0, 0, 0])
        p3 = np.array([0, 1, 0])
        angle = OrthopedicMeasurements.calculate_angle(p1, p2, p3)
        self.assertAlmostEqual(angle, 90.0)

    def test_cobb_angle(self):
        # Parallel lines
        l1s, l1e = np.array([0, 0, 0]), np.array([1, 0, 0])
        l2s, l2e = np.array([0, 1, 0]), np.array([1, 1, 0])
        angle = OrthopedicMeasurements.calculate_cobb_angle(l1s, l1e, l2s, l2e)
        self.assertAlmostEqual(angle, 0.0)

        # Perpendicular lines
        l2s, l2e = np.array([0, 0, 0]), np.array([0, 1, 0])
        angle = OrthopedicMeasurements.calculate_cobb_angle(l1s, l1e, l2s, l2e)
        self.assertAlmostEqual(angle, 90.0)

    def test_fracture_displacement(self):
        c1 = np.array([0, 0, 0])
        c2 = np.array([3, 4, 0])
        dist = FractureAnalyzer.measure_displacement(c1, c2)
        self.assertEqual(dist, 5.0)

if __name__ == "__main__":
    unittest.main()
