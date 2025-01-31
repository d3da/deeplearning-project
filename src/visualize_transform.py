import tqdm
import torch

from train import GTSRBDataset

import matplotlib.pyplot as plt
from torchvision import transforms

from ensemble import DefenseTransform


def to_numpy_image(input):
    return ((1 + input) / 2).detach().cpu().numpy().clip(0, 1).transpose(1,2,0)


def save_image(input, defense, save_path, noise_sigma =0.55, blur_kersize = 3, blur_sigma = 8.0):

    blurred_only = transforms.GaussianBlur(blur_kersize, blur_sigma).__call__(input)
    noisy = input + noise_sigma * torch.randn_like(input)
    noisy_and_blurred = transforms.GaussianBlur(blur_kersize, blur_sigma).__call__(noisy)

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for ax in axes.flat:
        ax.set(xticks=[], yticks=[])

    axes[0].set_title('Source image')
    axes[0].imshow(to_numpy_image(input))
    axes[1].set_title(f'Gaussian blur (kernel={blur_kersize}, σ={blur_sigma})')
    axes[1].imshow(to_numpy_image(blurred_only))
    
    axes[2].set_title(f'Gaussian noise (σ={noise_sigma})')
    axes[2].imshow(to_numpy_image(noisy))
    
    axes[3].set_title('Noise then Blurred')
    axes[3].imshow(to_numpy_image(noisy_and_blurred))

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

        if i > 8:
            break

        image_save_path = f'./transform_example/defense_transform_visualized_{i}.png'
        save_image(x, defense, image_save_path)


if __name__ == '__main__':
    main()
