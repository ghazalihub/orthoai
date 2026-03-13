import torch
from monai.networks.nets import SwinUNETR

def get_swin_unetr_model(in_channels=1, out_channels=2):
    """
    State-of-the-art Transformer-based 3D segmentation model.
    Swin UNETR uses a hierarchical transformer as an encoder.
    """
    model = SwinUNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=48,
        use_checkpoint=True,
    )
    return model

class AdvancedBoneSegmenter:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        self.model = get_swin_unetr_model().to(self.device)
        self.model.eval()

    def predict_with_uncertainty(self, x, num_samples=10):
        """
        Monte Carlo Dropout for Bayesian Uncertainty Quantification.
        """
        self.model.train() # Enable dropout
        samples = []
        with torch.no_grad():
            for _ in range(num_samples):
                samples.append(torch.softmax(self.model(x.to(self.device)), dim=1))

        samples = torch.stack(samples)
        mean_mask = samples.mean(dim=0)
        uncertainty = samples.std(dim=0)
        return mean_mask, uncertainty
