import tqdm
import torch

from train import GTSRBDataset

import matplotlib.pyplot as plt
from torchvision import transforms

from ensemble import DefenseTransform


def to_numpy_image(input):
    return ((1 + input) / 2).detach().cpu().numpy().clip(0, 1).transpose(1,2,0)


def save_image(input, defense, save_path):
    noisy = input + 0.25 * torch.randn_like(input)
    blurry = transforms.GaussianBlur(7, 2.5).__call__(noisy)

    fig, axes = plt.subplots(1, 3, figsize=(9, 4))
    for ax in axes.flat:
        ax.set(xticks=[], yticks=[])

    axes[0].set_title('Source image')
    axes[0].imshow(to_numpy_image(input))
    axes[1].set_title('+ Gaussian noise')
    axes[1].imshow(to_numpy_image(noisy))
    axes[2].set_title('+ Gaussian blur')
    axes[2].imshow(to_numpy_image(blurry))

    plt.tight_layout()

    plt.savefig(save_path)
    plt.close(fig)



def main():
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    defense = DefenseTransform()
    
    data_dir = "./data/GTSRB/Final_Training/Images/"
    val_csv = "./data/val.csv"
    val_dataset = GTSRBDataset(data_dir, val_csv, val_transform)


    for i, (x, y) in tqdm.tqdm(enumerate(val_dataset)):

        if i > 256:
            break

        image_save_path = f'./transform_example/defense_transform_visualized_{i}.png'
        save_image(x, defense, image_save_path)


if __name__ == '__main__':
    main()
