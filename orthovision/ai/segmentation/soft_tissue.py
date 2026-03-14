import torch
from monai.networks.nets import AttentionUnet

class SoftTissueSegmenter:
    """
    AI for segmenting soft tissue structures (ACL, PCL, Meniscus, Cartilage)
    from MRI sequences (T1, T2, Proton Density).
    """

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        self.model = AttentionUnet(
            spatial_dims=3,
            in_channels=3, # Multi-sequence input
            out_channels=5, # Background + 4 soft tissue classes
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
        ).to(self.device)

    def predict(self, mri_tensor: torch.Tensor) -> torch.Tensor:
        """
        Input tensor shape: (B, 3, H, W, D)
        """
        self.model.eval()
        with torch.no_grad():
            return torch.softmax(self.model(mri_tensor.to(self.device)), dim=1)
