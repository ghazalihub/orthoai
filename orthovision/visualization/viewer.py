import vtk

class OrthoViewer:
    """
    Skeleton for 3D visualization using VTK.
    """

    def __init__(self):
        self.renderer = vtk.vtkRenderer()
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)

    def add_mesh(self, stl_path: str, color: tuple = (0.8, 0.8, 0.8)):
        """
        Load and add an STL mesh to the viewer.
        """
        reader = vtk.vtkSTLReader()
        reader.SetFileName(stl_path)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(reader.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)

        self.renderer.AddActor(actor)

    def show(self):
        """
        Start the interactive viewer.
        """
        self.renderer.SetBackground(0.1, 0.2, 0.4) # Dark blue background
        self.render_window.Render()
        # self.interactor.Start() # Commented out for automated environments
        print("Viewer initialized. (Interactive mode disabled in sandbox)")
