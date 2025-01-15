import torch

from train import GTSRBDataset, clear_gpu_memory

def load_model(path):
    raise NotImplementedError



def iterative_fsgm(inputs, labels, model, loss_fn, step_size, num_steps):
    adversarial_inputs = inputs
    for i in range(num_steps):
        prediction = model(adversarial_inputs)
        loss = loss_fn(prediction, labels)
        gradient = torch.autograd.grad(loss, adversarial_inputs)
        gradient_sign = torch.sign(gradient)

        adversarial_inputs = adversarial_inputs + step_size * gradient_sign

    return adversarial_inputs


def accuracy(outputs, labels):
    _, predicted = outputs.max(1)
    val_total = labels.size(0)
    val_correct = predicted.eq(labels).sum().item()
    val_acc = 100.*val_correct/val_total

    return val_acc

def evaluate_model(model, dataloader, loss_fn, evaluation_metric):
    # 1. Evaluate accuracy with adversarial
    for inputs, labels in dataloader:
        adversarial_inputs = iterative_fsgm(inputs, labels, model, loss_fn)
        with torch.inference_mode():
            adv_predictions = model(adversarial_inputs)
            adv_evaluation = evaluation_metric(adv_predictions, labels)

    # 2. Evaluate accuracy without adversarial
    for inputs, labels in dataloader:
        with torch.inference_mode():
            normal_predictions = model(inputs)
            normal_evaluation = evaluation_metric(normal_predictions, labels)

    # 3. Create a dataframe or something with results
    # 4. Save to disk, and possibly some samples
    return adv_evaluation, normal_evaluation



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
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset paths
    data_dir = "/workspace/data/GTSRB/Final_Training/Images/"
    val_csv = "/workspace/data/val.csv"

    # Create datasets
    val_dataset = GTSRBDataset(data_dir, val_csv, val_transform)
    batch_size = 64

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    model_path = './data/model_checkpoint.pt'
    model = load_model(model_path)
    model.to(device)

    evaluate_model(model, val_loader, torch.nn.CrossEntropyLoss(), accuracy)
    

    
