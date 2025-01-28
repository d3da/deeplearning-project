import torch
import pandas as pd
import tqdm
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from train import GTSRBDataset, get_model
from adversarial import evaluate_model, accuracy, iterative_fsgm
from torch import nn as nn

# from torch import quantize_per_tensor

PATH_TO_ENSEMBLE = "/workspace/models/ensemble.pth"


class EnsembleModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
        # returns <10% after 1 step without defense

    def forward(self, x):
        logits = [model(x) for model in self.models]
        return torch.mean(torch.stack(logits), dim=0)

class DefenseTransform(nn.Module):
    def __init__(self, kernel_size=7, noise_std=0.25):
        super().__init__()
        self.noise_std = noise_std
        self.gaussian_blur = transforms.GaussianBlur(kernel_size, sigma=2.5)
        
    def forward(self, x):
        # adding in gaussian noise
        if self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        
        x = self.gaussian_blur(x)
        
        x = torch.clamp(x, -2.5, 2.5)  # kinda works with the val_transform
        return x





def load_individual_models(device):
    """Load the three base models and put them in eval mode"""
    models = []
    model_paths = [
        "./models/final_regnet_e10_b512_lr1e-03_20250125_0022.pth",
        "./models/final_maxvit_e10_b64_lr1e-03_20250125_0039.pth",
        "./models/final_efficientnet_e10_b64_lr1e-03_20250125_0058.pth",
    ]

    num_classes = 43
    for path in model_paths:
        # Initialize model architecture
        if "regnet" in path:
            model = get_model("regnet", num_classes=num_classes)
        elif "maxvit" in path:
            model = get_model("maxvit", num_classes=num_classes)
        elif "efficientnet" in path:
            model = get_model("efficientnet", num_classes=num_classes)

        checkpoint = torch.load(path, map_location=device)

        # Extract model state_dict
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Fallback: Assume the file is just the state_dict
            model.load_state_dict(checkpoint)

        # Configure model
        model.eval()
        model.to(device)
        models.append(model)

    return models


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        print(
            f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )

    # Load models and create ensemble
    models = load_individual_models(device)
    # ensemble = EnsembleModel(models)
    ensemble = nn.Sequential(DefenseTransform(), EnsembleModel(models))

    ensemble.eval()
    ensemble.to(device)

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # still crashes on 24gb 4090, but does fit with headroom on 32 batch size
    current_batch_size = 64
    success = False

    while not success:
        try:
            data_dir = "./data/GTSRB/Final_Training/Images/"
            val_csv = "./data/val.csv"
            val_dataset = GTSRBDataset(data_dir, val_csv, val_transform)

            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=current_batch_size,  # Use dynamic batch size
                shuffle=False,
                num_workers=2,
                pin_memory=True,
                persistent_workers=True,
            )

            image_save_dir = "./samples/ensemble"
            os.makedirs(image_save_dir, exist_ok=True)

            results_df = evaluate_model(
                model=ensemble,
                dataloader=val_loader,
                loss_fn=nn.CrossEntropyLoss(),
                evaluation_metric=accuracy,
                image_save_dir=image_save_dir,
            )

            print(f"\n{'-'*40}")
            print("Ensemble Results:")
            print(f"{'-'*40}")
            print(f"Normal accuracy: {results_df['normal_accuracy'].mean():.2f}%")
            print(f"Adversarial accuracy: {results_df['adversarial_accuracy'].mean():.2f}%")

            success = True

        except RuntimeError as e:
            if "out of memory" in str(e):
                # Cleanup and retry with smaller batch size
                del ensemble, val_loader, val_dataset
                torch.cuda.empty_cache()

                current_batch_size = current_batch_size // 2
                print(f"\nOOM error: Reduced batch size to {current_batch_size}")

                if current_batch_size < 8:
                    print("ERROR: Batch size too small. Aborting.")
                    break

                # Recreate models and ensemble
                models = load_individual_models(device)
                ensemble = EnsembleModel(models).to(device)
                ensemble.eval()
            else:
                print(f"Error evaluating ensemble: {str(e)}")
                break


if __name__ == "__main__":
    main()

