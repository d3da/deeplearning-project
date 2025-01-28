import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms, utils
import os

from train import GTSRBDataset, get_model
from adversarial import evaluate_model, accuracy, iterative_fsgm, load_model
from ensemble import EnsembleModel, DefenseTransform, load_individual_models

def eval_model_defense(model, dataloader, results_dir, name):
    """Test without and with defense"""

    no_defense_dir = os.path.join(results_dir, 'no_defense')
    defense_dir = os.path.join(results_dir, 'defense')

    eval_model(model, dataloader, no_defense_dir, name=f'{name} (no defense)')
    eval_model(nn.Sequential(DefenseTransform(), model), dataloader, defense_dir, name=f'{name} (with defense)')


def eval_model(model, dataloader, results_dir, name):
    os.makedirs(results_dir, exist_ok=True)
    results_df = evaluate_model(
        model=model,
        dataloader=dataloader,
        loss_fn=nn.CrossEntropyLoss(),
        evaluation_metric=accuracy,
        image_save_dir=results_dir
    )

    print(f"\n{'-'*40}")
    print(f"Results for {name}:")
    print(f"{'-'*40}")
    print(f"Normal accuracy: {results_df['normal_accuracy'].mean():.2f}%")
    print(f"Adv. accuracy: {results_df['adversarial_accuracy'].mean():.2f}%")

    results_df.to_csv(os.path.join(results_dir, 'dataframe.csv'))



def eval_validation_test(model_configs, val_dataloader, test_dataloader, base_results_dir, device):

    # 1. Evaluate the base models
    for model_key, config in model_configs.items():
        model_path = config["path"]
        model_name = config["name"]
        num_classes = 43
        model = load_model(model_path, model_name, num_classes, device)
        model.eval()

        validation_results_dir = os.path.join(base_results_dir, 'validation', model_name)
        test_results_dir = os.path.join(base_results_dir, 'test', model_name)
        eval_model_defense(model, val_dataloader, validation_results_dir, model_name)
        eval_model_defense(model, test_dataloader, test_results_dir, model_name)

        del model
        torch.cuda.empty_cache()


    # 2. Evaluate the ensemble
    model = EnsembleModel(load_individual_models(device))
    model.eval()
    model.to(device)
    validation_results_dir = os.path.join(base_results_dir, 'validation/ensemble')
    test_results_dir = os.path.join(base_results_dir, 'test/ensemble')
    eval_model_defense(model, val_dataloader, validation_results_dir, 'ensemble')
    eval_model_defense(model, test_dataloader, test_results_dir, 'ensemble')


def main():
    # Set device and optimize CUDA if available
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Training will be very slow on CPU!")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        print(
            f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )

    data_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    MODEL_CONFIGS = {
        "regnet": {
            "path": "./models/final_regnet_e10_b512_lr1e-03_20250125_0022.pth",
            "batch_size": 512,
            "name": "regnet",
        },
        "maxvit": {
            "path": "./models/final_maxvit_e10_b64_lr1e-03_20250125_0039.pth",
            "batch_size": 128,
            "name": "maxvit",
        },
        "efficientnet": {
            "path": "./models/final_efficientnet_e10_b64_lr1e-03_20250125_0058.pth",
            "batch_size": 128,
            "name": "efficientnet",
        },
    }

    val_data_dir = "./data/GTSRB/Final_Training/Images/"
    val_csv = "./data/val.csv"
    val_dataset = GTSRBDataset(val_data_dir, val_csv, data_transform)

    test_data_dir = "./data/GTSRB/Final_Test/Images/"
    test_csv = "./data/test.csv"
    test_dataset = GTSRBDataset(test_data_dir, test_csv, data_transform)

    base_results_dir = "./final_experiment"
    batch_size = 32

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    eval_validation_test(MODEL_CONFIGS, val_loader, test_loader, base_results_dir, device)


if __name__ == "__main__":
    main()

