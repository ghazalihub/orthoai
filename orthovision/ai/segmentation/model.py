import torch
from monai.networks.nets import UNet
from monai.networks.layers import Norm

def get_bone_segmentation_model(in_channels=1, out_channels=2):
    """
    Returns a MONAI UNet model configured for bone segmentation.
    Default: 1 input channel (CT/X-ray), 2 output channels (Background, Bone).
    """
    model = UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    )
    return model

class BoneSegmenter:
    def __init__(self, model_path=None, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        self.model = get_bone_segmentation_model().to(self.device)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def predict(self, image_tensor):
        """
        image_tensor: torch.Tensor of shape (B, C, H, W, D)
        """
        with torch.no_grad():
            output = self.model(image_tensor.to(self.device))
            return torch.softmax(output, dim=1)
