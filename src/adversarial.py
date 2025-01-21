import pandas as pd
import tqdm
import torch
import os
import sys
from torchvision import transforms, utils

from train import GTSRBDataset, clear_gpu_memory, get_model

STEP_SIZE = 5e-2
NUM_STEPS = 5


def load_model(path, model_name, num_classes, device):

    model = get_model(model_name, num_classes)
    state_dict = torch.load(path, map_location=device)["model_state_dict"]
    model.load_state_dict(state_dict)

    return model.to(device)


def iterative_fsgm(
    inputs, labels, model, loss_fn, evaluation_metric, step_size=1e-2, num_steps=1, index_info=None, image_save_dir=None, image_save_freq=20
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
        save_image(input_index, image_save_dir, inputs, adversarial_inputs, initial_pred, prediction, labels)

    return adversarial_inputs, normal_metric, adversarial_metric


def save_image(input_index, image_save_dir, inputs, adversarial_inputs, initial_pred, adv_pred, labels):
    image_path = os.path.join(image_save_dir, f'inputs_{input_index}_normal.png')
    adv_image_path = os.path.join(image_save_dir, f'inputs_{input_index}_adversarial.png')

    utils.save_image(inputs, image_path)
    utils.save_image(adversarial_inputs, adv_image_path)


def accuracy(outputs, labels):
    _, predicted = outputs.max(1)
    val_total = labels.size(0)
    val_correct = predicted.eq(labels).sum().item()
    val_acc = 100.0 * val_correct / val_total

    return val_acc


def evaluate_model(model, dataloader, loss_fn, evaluation_metric, image_save_dir=None, image_save_freq=20) -> pd.DataFrame:
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

    model_path = "./models/final_regnet_e10_b128_lr1e-03_20250107_0233.pth"
    model_name = "regnet"
    num_classes = 43

    model = load_model(model_path, model_name, num_classes, device)

    # Dataset paths
    data_dir = "./data/GTSRB/Final_Training/Images/"
    val_csv = "./data/val.csv"

    # Create datasets
    val_dataset = GTSRBDataset(data_dir, val_csv, val_transform)
    batch_size = 128

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )


    image_save_dir = f'./samples/{model_name}'
    os.makedirs(image_save_dir, exist_ok=True)

    results_df = evaluate_model(model, val_loader, torch.nn.CrossEntropyLoss(), accuracy, image_save_dir=image_save_dir)
    print(results_df)
    print(f"Normal accuracy: {results_df['normal_accuracy'].mean():.2f}%")
    print(f"Adversarial accuracy: {results_df['adversarial_accuracy'].mean():.2f}%")

    # pdb.set_trace()


if __name__ == "__main__":
    main()
