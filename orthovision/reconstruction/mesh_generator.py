import numpy as np
from skimage import measure
import vtk
from vtk.util import numpy_support

class MeshGenerator:
    """
    Generates 3D meshes from segmentation masks.
    """

    @staticmethod
    def generate_mesh_marching_cubes(mask: np.ndarray, spacing: tuple = (1.0, 1.0, 1.0)) -> tuple:
        """
        Generate mesh using Marching Cubes algorithm.
        Returns: verts, faces, normals, values
        """
        verts, faces, normals, values = measure.marching_cubes(mask, level=0.5, spacing=spacing)
        return verts, faces, normals, values

    @staticmethod
    def save_as_stl(verts: np.ndarray, faces: np.ndarray, filename: str):
        """
        Save the mesh as an STL file using VTK.
        """
        points = vtk.vtkPoints()
        for v in verts:
            points.InsertNextPoint(v)

        polys = vtk.vtkCellArray()
        for f in faces:
            triangle = vtk.vtkTriangle()
            triangle.GetPointIds().SetId(0, f[0])
            triangle.GetPointIds().SetId(1, f[1])
            triangle.GetPointIds().SetId(2, f[2])
            polys.InsertNextCell(triangle)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(polys)

        # Optional: smoothing
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputData(polydata)
        smoother.SetNumberOfIterations(15)
        smoother.BoundarySmoothingOn()
        smoother.FeatureEdgeSmoothingOn()
        smoother.Update()

        writer = vtk.vtkSTLWriter()
        writer.SetFileName(filename)
        writer.SetInputData(smoother.GetOutput())
        writer.Write()
