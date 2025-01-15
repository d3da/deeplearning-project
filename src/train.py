import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import timm
from PIL import Image
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import datetime
import time
import datetime
import torch.nn.functional as F
import gc



class GTSRBDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.data.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def get_model(model_name, num_classes):
    if model_name == "regnet":
        model = timm.create_model('regnety_008', pretrained=True, num_classes=num_classes)
        if hasattr(model, 'head') and hasattr(model.head, 'fc'):
            in_features = model.head.fc.in_features
            model.head.fc = nn.Linear(in_features, num_classes)
    elif model_name == "maxvit":
        model = timm.create_model('maxvit_tiny_tf_224', pretrained=True, num_classes=num_classes)
        if hasattr(model, 'head') and hasattr(model.head, 'fc'):
            in_features = model.head.fc.in_features
            model.head.fc = nn.Linear(in_features, num_classes)
    elif model_name == "efficientnet":
        model = timm.create_model('tf_efficientnetv2_m', pretrained=True, num_classes=num_classes)
        if hasattr(model, 'classifier'):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
    
    print(f"Created {model_name} model with architecture:")
    print(model)
    return model

def train_model(model_name, model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_save_name):
    best_acc = 0.0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\n{'-'*20} {model_name.upper()} - EPOCH {epoch+1}/{num_epochs} {'-'*20}")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training loop with progress bar
        pbar = tqdm(train_loader, desc=f'{model_name.upper()} Epoch {epoch+1}/{num_epochs}', 
            dynamic_ncols=True, file=sys.stdout, leave=True)


        batch_start_time = time.time()
        
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Calculate speeds and ETAs
            images_per_sec = inputs.size(0) / (time.time() - batch_start_time)
            epoch_progress = (batch_idx + 1) / len(train_loader)
            epoch_remaining = (time.time() - start_time) * (1/epoch_progress - 1)
            total_progress = (epoch + epoch_progress) / num_epochs
            training_remaining = (time.time() - start_time) * (1/total_progress - 1)
            
            # GPU memory usage
            gpu_memory_used = torch.cuda.memory_reserved(device) / 1024**3
            gpu_memory_total = torch.cuda.get_device_properties(device).total_memory / 1024**3
            
            pbar.set_postfix({
                'Loss': f"{running_loss/(batch_idx+1):.3f}",
                'Acc': f"{100.*correct/total:.1f}%",
                'img/s': f"{images_per_sec:.1f}",
                'GPU': f"{gpu_memory_used:.1f}/{gpu_memory_total:.1f}GB",
                'ETA e': f"{datetime.timedelta(seconds=int(epoch_remaining))}",
                'ETA t': f"{datetime.timedelta(seconds=int(training_remaining))}"
            })
            
            batch_start_time = time.time()
        
        # Clear memory after training loop
        clear_gpu_memory()
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        print(f"\nRunning {model_name.upper()} validation...")
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'{model_name.upper()} Validation'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100.*val_correct/val_total
        epoch_time = time.time() - start_time
        
        print(f'\n{model_name.upper()} Epoch Summary:')
        print(f'Training Loss: {running_loss/len(train_loader):.3f}')
        print(f'Training Accuracy: {100.*correct/total:.2f}%')
        print(f'Validation Loss: {val_loss/len(val_loader):.3f}')
        print(f'Validation Accuracy: {val_acc:.2f}%')
        print(f'Epoch Time: {datetime.timedelta(seconds=int(epoch_time))}')
        print(f'GPU Memory: {gpu_memory_used:.1f}/{gpu_memory_total:.1f}GB')
        
        if val_acc > best_acc:
            best_acc = val_acc
            print(f'New best {model_name.upper()} model! Validation Accuracy: {val_acc:.2f}%')
            # Save best model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': 100.*correct/total,
            }, f'/workspace/models/best_{model_save_name}.pth')
        
        # Clear memory after validation
        clear_gpu_memory()
    
    return model

def save_checkpoint(model_name, model, optimizer, scheduler, epoch, model_save_name, is_best=False):
    """Save model checkpoint"""
    save_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    
    prefix = "best" if is_best else "final"
    path = f'/workspace/models/{prefix}_{model_save_name}.pth'
    torch.save(save_data, path)
    print(f"Saved {model_name.upper()} {prefix} model to {path}")

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
    
    # Data augmentation and normalization
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset paths
    data_dir = "/workspace/data/GTSRB/Final_Training/Images/"
    train_csv = "/workspace/data/train.csv"
    val_csv = "/workspace/data/val.csv"
    
    # Create datasets
    train_dataset = GTSRBDataset(data_dir, train_csv, train_transform)
    val_dataset = GTSRBDataset(data_dir, val_csv, val_transform)
    
    # Initial batch sizes
    batch_sizes = {
        "regnet": 128,
        "maxvit": 64,
        "efficientnet": 64
    }
    val_batch_size_hi = 128
    val_batch_size_lo = 32
    
    # Training parameters
    num_classes = 43
    num_epochs = 10
    learning_rate = 0.001
    
    # Train each model
    # models = ["maxvit","efficientnet"]
    models = ["regnet","efficientnet","maxvit"]

    
    for model_name in models:
        print(f"\n{'-'*40}")
        print(f"Training {model_name.upper()}")
        print(f"{'-'*40}")
        if model_name == "maxvit":
            val_batch_size = val_batch_size_lo
        else:
            val_batch_size = val_batch_size_hi
        
        while True:
            try:
                # Clear memory before starting new model
                clear_gpu_memory()
                
                # Create data loaders with current batch size
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=batch_sizes[model_name],
                    shuffle=True,
                    num_workers=2,
                    pin_memory=True,
                    persistent_workers=True
                )
                
                val_loader = DataLoader(
                    val_dataset, 
                    batch_size=val_batch_size,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=True,
                    persistent_workers=True
                )
                
                # Create model and move to GPU
                model = get_model(model_name, num_classes)
                model = model.to(device)
                print(f"{model_name.upper()} created and moved to {device}")
                print(f"Current batch size: {batch_sizes[model_name]}")
                
                # Create model save name
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                model_save_name = (
                    f"{model_name}_"
                    f"e{num_epochs}_"
                    f"b{batch_sizes[model_name]}_"
                    f"lr{learning_rate:.0e}_"
                    f"{timestamp}"
                )
                
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='max', factor=0.1, patience=3
                )
                
                model = train_model(
                    model_name=model_name,
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    num_epochs=num_epochs,
                    device=device,
                    model_save_name=model_save_name
                )
                
                # Save final checkpoint
                save_checkpoint(model_name, model, optimizer, scheduler, num_epochs, 
                              model_save_name, is_best=False)
                
                # If we get here, training was successful
                break
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # Clear memory
                    clear_gpu_memory()
                    
                    # Reduce batch size for this model
                    batch_sizes[model_name] = batch_sizes[model_name] // 2
                    print(f"\nCUDA OOM error. Reducing {model_name.upper()} batch size to {batch_sizes[model_name]}")
                    
                    if batch_sizes[model_name] < 8:
                        print(f"ERROR: Batch size for {model_name.upper()} too small. Skipping model.")
                        break
                    
                    # Try again with new batch size
                    continue
                else:
                    print(f"Error during {model_name.upper()} training: {str(e)}")
                    break
            except Exception as e:
                print(f"Error during {model_name.upper()} training: {str(e)}")
                break

if __name__ == "__main__":
    main()
