import pandas as pd
import tqdm
import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, utils

from train import GTSRBDataset, get_model

STEP_SIZE = 9e-3
NUM_STEPS = 10

gtsrb_label_map = {
    0: "20_speed",
    1: "30_speed",
    2: "50_speed",
    3: "60_speed",
    4: "70_speed",
    5: "80_speed",
    6: "80_lifted",
    7: "100_speed",
    8: "120_speed",
    9: "no_overtaking_general",
    10: "no_overtaking_trucks",
    11: "right_of_way_crossing",
    12: "right_of_way_general",
    13: "give_way",
    14: "stop",
    15: "no_way_general",
    16: "no_way_trucks",
    17: "no_way_one_way",
    18: "attention_general",
    19: "attention_left_turn",
    20: "attention_right_turn",
    21: "attention_curvy",
    22: "attention_bumpers",
    23: "attention_slippery",
    24: "attention_bottleneck",
    25: "attention_construction",
    26: "attention_traffic_light",
    27: "attention_pedestrian",
    28: "attention_children",
    29: "attention_bikes",
    30: "attention_snowflake",
    31: "attention_deer",
    32: "lifted_general",
    33: "turn_right",
    34: "turn_left",
    35: "turn_straight",
    36: "turn_straight_right",
    37: "turn_straight_left",
    38: "turn_right_down",
    39: "turn_left_down",
    40: "turn_circle",
    41: "lifted_no_overtaking_general",
    42: "lifted_no_overtaking_trucks",
}


def load_model(path, model_name, num_classes, device):

    model = get_model(model_name, num_classes)
    state_dict = torch.load(path, map_location=device)["model_state_dict"]
    model.load_state_dict(state_dict)

    return model.to(device)


def iterative_fsgm(
    inputs,
    labels,
    model,
    loss_fn,
    evaluation_metric,
    step_size=1e-2,
    num_steps=1,
    index_info=None,
    image_save_dir=None,
    image_save_freq=20,
):
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    labels = labels.to(device)
    adversarial_inputs = inputs.requires_grad_(True)

    # load once to get initial accuracy for prints
    with torch.inference_mode():
        initial_pred = model(inputs)
        initial_acc = evaluation_metric(initial_pred, labels)

    if index_info is not None:
        input_index, input_len = index_info
        desc = f"{input_index}/{input_len} FGSM step:"

    pbar = tqdm.tqdm(range(num_steps), desc=desc, dynamic_ncols=True, file=sys.stdout, leave=True)
    for i in pbar:
        prediction = model(adversarial_inputs)
        loss = loss_fn(prediction, labels)
        current_acc = evaluation_metric(prediction, labels)

        if i == 0:
            normal_metric = evaluation_metric(prediction, labels)

        gradient = torch.autograd.grad(loss, adversarial_inputs)[0]
        gradient_sign = torch.sign(gradient)

        adversarial_inputs = adversarial_inputs + step_size * gradient_sign
        adversarial_inputs.clamp(0, 1)

        acc_drop = initial_acc - current_acc
        # progress bar
        pbar.set_postfix(
            {
                "Loss": f"{loss.item():.3f}",
                "Curr Acc": f"{current_acc:.1f}%",
                "Acc Drop": f"{acc_drop:.1f}%",
                "Step": f"{step_size:.3f}",
            }
        )

    with torch.inference_mode():
        prediction = model(adversarial_inputs)
        adversarial_metric = evaluation_metric(prediction, labels)

    if (image_save_dir is not None) and (input_index % image_save_freq == 0):
        _, predicted_labels = initial_pred.max(1)
        _, adv_predicted_labels = prediction.max(1)
        save_image(
            input_index,
            image_save_dir,
            inputs,
            adversarial_inputs,
            predicted_labels,
            adv_predicted_labels,
            labels,
        )
        save_image_plt(
            input_index,
            image_save_dir,
            inputs,
            adversarial_inputs,
            predicted_labels,
            adv_predicted_labels,
            labels,
        )

    return adversarial_inputs, normal_metric, adversarial_metric


def save_image(
    input_index, image_save_dir, inputs, adversarial_inputs, initial_pred, adv_pred, labels
):
    image_path = os.path.join(image_save_dir, f"inputs_{input_index}_normal.png")
    adv_image_path = os.path.join(image_save_dir, f"inputs_{input_index}_adversarial.png")

    utils.save_image(inputs, image_path)
    utils.save_image(adversarial_inputs, adv_image_path)


def save_image_plt(
    input_index,
    image_save_dir,
    inputs,
    adversarial_inputs,
    initial_pred,
    adv_pred,
    labels,
    max_images=5,
):
    inputs = inputs[:max_images]
    adversarial_inputs = adversarial_inputs[:max_images]
    initial_pred = initial_pred[:max_images].detach().cpu().numpy()
    adv_pred = adv_pred[:max_images].detach().cpu().numpy()
    labels = labels[:max_images].detach().cpu().numpy()
    inputs = inputs.detach().cpu().numpy().transpose(0, 2, 3, 1)
    adversarial_inputs = adversarial_inputs.detach().cpu().numpy().transpose(0, 2, 3, 1)

    fig, axes = plt.subplots(2, len(inputs), figsize=(15, 6))
    for ax in axes.flat:
        ax.set(xticks=[], yticks=[])

    for i in range(max(len(inputs), max_images)):
        axes[0, i].imshow(np.clip(inputs[i], 0, 1))
        axes[0, i].set_title(
            f"True label: {gtsrb_label_map[labels[i]]}\n"
            f"Normal pred: {gtsrb_label_map[initial_pred[i]]}"
        )

        axes[1, i].imshow(np.clip(adversarial_inputs[i], 0, 1))
        axes[1, i].set_title(f"Adv pred: {gtsrb_label_map[adv_pred[i]]}")

    # Adjust layout and save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    grid_image_path = os.path.join(image_save_dir, f"inputs_{input_index}_grid.png")
    plt.savefig(grid_image_path)
    plt.close(fig)


def accuracy(outputs, labels):
    _, predicted = outputs.max(1)
    val_total = labels.size(0)
    val_correct = predicted.eq(labels).sum().item()
    val_acc = 100.0 * val_correct / val_total

    return val_acc


def evaluate_model(
    model, dataloader, loss_fn, evaluation_metric, image_save_dir=None, image_save_freq=20
) -> pd.DataFrame:
    normal_metrics = []
    adversarial_metrics = []
    input_len = len(dataloader)
    for input_index, (inputs, labels) in enumerate(tqdm.tqdm(dataloader)):
        adv_inputs, normal_metric, adversarial_metric = iterative_fsgm(
            inputs,
            labels,
            model,
            loss_fn,
            evaluation_metric,
            STEP_SIZE,
            NUM_STEPS,
            index_info=(input_index, input_len),
            image_save_dir=image_save_dir,
            image_save_freq=image_save_freq,
        )
        # adversarial_inputs.append(adv_inputs)
        normal_metrics.append(normal_metric)
        adversarial_metrics.append(adversarial_metric)

    results_df = pd.DataFrame(
        {"normal_accuracy": normal_metrics, "adversarial_accuracy": adversarial_metrics}
    )
    return results_df


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

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    #
    # MODEL_CONFIGS = {
    #     "regnet": {
    #         "path": "./models/final_regnet_e10_b128_lr1e-03_20250107_0233.pth",
    #         "batch_size": 512,
    #         "name": "regnet",
    #     },
    #     "maxvit": {
    #         "path": "./models/final_maxvit_e10_b16_lr1e-03_20250107_0527.pth",
    #         "batch_size": 128,
    #         "name": "maxvit",
    #     },
    #     "efficientnet": {
    #         "path": "./models/final_efficientnet_e10_b16_lr1e-03_20250107_0335.pth",
    #         "batch_size": 128,
    #         "name": "efficientnet",
    #     },
    # }
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

    # Dataset paths
    data_dir = "./data/GTSRB/Final_Training/Images/"
    val_csv = "./data/val.csv"
    val_dataset = GTSRBDataset(data_dir, val_csv, val_transform)

    # Iterate through each model
    for model_key in MODEL_CONFIGS:
        config = MODEL_CONFIGS[model_key]
        model_path = config["path"]
        model_name = config["name"]
        original_batch_size = config["batch_size"]
        current_batch_size = original_batch_size
        num_classes = 43

        success = False
        while True:
            try:
                # Load model
                model = load_model(model_path, model_name, num_classes, device)

                # Create data loader with current batch size
                val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=current_batch_size,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=True,
                    persistent_workers=True,
                )

                image_save_dir = f"./samples/{model_name}"
                os.makedirs(image_save_dir, exist_ok=True)

                # Evaluate model
                #
                results_df = evaluate_model(
                    model,
                    val_loader,
                    torch.nn.CrossEntropyLoss(),
                    accuracy,
                    image_save_dir=image_save_dir,
                )

                print(f"\n{'-'*40}")
                print(f"Results for {model_name.upper()}:")
                print(f"{'-'*40}")
                print(results_df)
                print(f"Normal accuracy: {results_df['normal_accuracy'].mean():.2f}%")
                print(f"Adversarial accuracy: {results_df['adversarial_accuracy'].mean():.2f}%")

                success = True
                break  # Exit loop on success

            except RuntimeError as e:
                if "out of memory" in str(e):
                    # Cleanup and retry with reduced batch size
                    del model
                    del val_loader
                    torch.cuda.empty_cache()

                    current_batch_size = current_batch_size // 2
                    print(
                        f"\nOOM error: Reduced {model_name.upper()} batch size to {current_batch_size}"
                    )

                    if current_batch_size < 8:
                        print(f"ERROR: Batch size too small for {model_name.upper()}. Skipping.")
                        break
                else:
                    print(f"Error evaluating {model_name.upper()}: {str(e)}")
                    break
            except Exception as e:
                print(f"Unexpected error evaluating {model_name.upper()}: {str(e)}")
                break
            finally:
                # Cleanup if not successful
                if not success:
                    if "model" in locals():
                        del model
                    if "val_loader" in locals():
                        del val_loader
                    torch.cuda.empty_cache()

        # Final cleanup for current model
        if success:
            del model
            del val_loader
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
