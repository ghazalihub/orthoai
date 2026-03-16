import torch

class FederatedOrthoClient:
    """
    Client-side logic for Federated Learning (FL).
    Enables multi-institutional model training without sharing patient images.
    """

    def __init__(self, local_model: torch.nn.Module):
        self.model = local_model

    def train_local_epoch(self, data_loader, optimizer, criterion):
        """
        Trains the model on local institutional data.
        """
        self.model.train()
        for images, labels in data_loader:
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    def get_model_parameters(self) -> dict:
        """
        Returns model weights for aggregation by the central server.
        """
        return self.model.state_dict()

    def update_model(self, global_parameters: dict):
        """
        Updates the local model with aggregated weights from the federation.
        """
        self.model.load_state_dict(global_parameters)
