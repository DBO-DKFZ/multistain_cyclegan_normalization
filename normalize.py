import os

import PIL.Image
import torch

import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from pathlib import Path
from PIL import Image
from models.networks import UnetGenerator, get_norm_layer
from util.util import tensor2im, tensor2im_batch
import torch
from torch.utils.data import Dataset
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, img_list, transform):
        self.img_list = img_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        file_name = img_path.name
        image = Image.open(img_path).convert('RGB')
        normalized = self.transform(image)
        return normalized, file_name


def main():
    cwd = Path.cwd()
    script_path = cwd.resolve()

    model_path = os.path.join(script_path, 'resources', 'models', 'latest_net_G_A.pth')

    print("Creating model")
    model = UnetGenerator(3, 3, 8, ngf=64, norm_layer=get_norm_layer('instance'), use_dropout=False)
    model.load_state_dict(torch.load(model_path))

    center_list = os.listdir(os.path.join(script_path, 'resources', 'test_data'))

    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])

    for center in center_list:
        image_list = os.listdir(os.path.join(script_path, 'resources', 'test_data', center))

        if center == 'center_0':
            fig, ax = plt.subplots(1, len(image_list), sharex=True, sharey=True)
            fig.suptitle('Source Domain Images', fontsize=16)
        else:
            fig, ax = plt.subplots(2, len(image_list), sharex=True, sharey=True)
            fig.suptitle(center + '$\Longrightarrow$ ' + 'center_0', fontsize=16)

        for i, image in enumerate(image_list):
            image_path = os.path.join(script_path, 'resources', 'test_data', center, image)
            img = Image.open(image_path).convert('RGB')

            if center == 'center_0':
                ax[i].imshow(img)

            else:
                normalized = transform(img)
                transformed = normalized[None, :]

                output = model(transformed).detach()
                out_img = tensor2im(output)

                ax[0, i].imshow(img)
                ax[1, i].imshow(out_img)
                ax[0, 0].set_ylabel("Unnormalized", fontsize=16)
                ax[1, 0].set_ylabel("Normalized", fontsize=16)

        plt.tight_layout()
        plt.show()
    print("finished!")


if __name__ == "__main__":
    '''
    Minimal working example of the normalization with a given Generator
    '''
    main()
