import open3d as o3d
import numpy as np
import os
from typing import Tuple, List

class ImplantFitter:
    """
    Automated implant selection and registration using ICP and Global Registration.
    """

    def __init__(self, implant_library_path: str = "assets/implants"):
        self.library_path = implant_library_path

    @staticmethod
    def preprocess_point_cloud(pcd, voxel_size):
        res = pcd.voxel_down_sample(voxel_size)
        res.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            res, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
        return res, fpfh

    def auto_fit(self, bone_pcd: o3d.geometry.PointCloud,
                 implant_pcd: o3d.geometry.PointCloud,
                 voxel_size: float = 2.0) -> Tuple[np.ndarray, float]:
        """
        End-to-end global + local (ICP) registration.
        """
        source_down, source_fpfh = self.preprocess_point_cloud(implant_pcd, voxel_size)
        target_down, target_fpfh = self.preprocess_point_cloud(bone_pcd, voxel_size)

        # 1. Global RANSAC Registration
        distance_threshold = voxel_size * 1.5
        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))

        # 2. Local Refinement (Point-to-Plane ICP)
        # Ensure normals are estimated for point-to-plane
        bone_pcd.estimate_normals()
        result_icp = o3d.pipelines.registration.registration_icp(
            implant_pcd, bone_pcd, distance_threshold, result_ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())

        return result_icp.transformation, result_icp.fitness

    def predict_size(self, bone_geometry_features: dict) -> str:
        """
        Placeholder for ML-based size prediction based on bone morphology.
        """
        # Logic: If femur_width > X and neck_length > Y -> Large
        return "Medium-14"
