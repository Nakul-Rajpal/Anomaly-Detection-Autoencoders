# Nakul Rajpal 
# Start 6/10/2025
# For NDSU AI SUSTEIN REU 2025 

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from data_loading import get_data
from train import train

print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0))

# Autoencoder Class For Training
class Autoencoder(nn.Module):
    # Added input dimension to the constructor: number of features in the dataset
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        # Encoder: Reduces input dimension to 8 features
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )

        # Decoder: Reconstructs input from 8 features back to original dimension
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def main():
    train_loader, val_loader, test_loader, input_dim = get_data()
    model = Autoencoder(input_dim)
    
    trained_model = train(
        model = model,
        dataloader = train_loader,
        num_epochs = 50, 
        lr=0.001
    )

    

if __name__ == "__main__":
    main()







