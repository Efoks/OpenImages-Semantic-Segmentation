import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image
import matplotlib.pyplot as plt
import src.config as cfg
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torchvision.transforms as transforms


class OpenImagesDataset(Dataset):

    def __init__(self, data_dir, classes, class_suffixes = ['m0663v', 'm0pg52', 'm07clx'], transform=None, normalize=None):
        self.data_dir = data_dir
        self.transform = transform
        self.normalize = normalize
        self.class_suffixes = class_suffixes
        self.images = []
        self.masks = []
        self.classes_dir = []
        for cls in classes:

            class_dir = os.path.join(data_dir, cls.lower())
            images_dir = os.path.join(class_dir, 'images')
            masks_dir = os.path.join(class_dir, 'segmentations')

            self.classes_dir.append(class_dir)
            images = os.listdir(images_dir)
            masks = os.listdir(masks_dir)

            for image in images:
                pre_mask_name = image.split('.')[0]
                mask_name = next((name for name in masks if name.startswith(pre_mask_name)), None)
                if mask_name is not None:
                    self.images.append(image)
                    self.masks.append(mask_name)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        pre_mask_name = image_name.split('.')[0]
        mask_name = next((name for name in self.masks if name.startswith(pre_mask_name)), None)

        if mask_name is None:
            raise ValueError(f'No mask found for image {image_name}')

        for class_dir in self.classes_dir:
            if os.path.exists(os.path.join(class_dir, 'segmentations', mask_name)):
                mask_path = os.path.join(class_dir, 'segmentations', mask_name)
                class_name = os.path.basename(class_dir)

            if os.path.exists(os.path.join(class_dir, 'images', image_name)):
                img_path = os.path.join(class_dir, 'images', image_name)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
        if self.normalize:
            img = self.normalize(img)

        mask = torch.where(mask == 1, torch.tensor(cfg.CLASS_DICT[class_name]), torch.tensor(0))
        mask = F.one_hot(mask.to(torch.int64).squeeze(), num_classes=4)
        mask = mask.permute(2, 0, 1)

        return img, mask

def create_data_split(images_idx, masks_idx):
    images_train, images_test, masks_train, masks_test = train_test_split(images_idx, masks_idx, test_size=0.2, random_state=42)
    train_sampler = SubsetRandomSampler(images_train)
    test_sampler = SubsetRandomSampler(images_test)
    return train_sampler, test_sampler

def create_data_loader(batch_size, data_dir = cfg.DATA_DIR, classes = ['Pizza', 'Taxi', 'Dog'], transform=None, normalize=None):
    dataset = OpenImagesDataset(data_dir, classes, transform=transform, normalize=normalize)
    train_sampler, test_sampler = create_data_split(range(len(dataset)), range(len(dataset)))

    train_data_loader = DataLoader(dataset, batch_size=batch_size, sampler = train_sampler)
    test_data_loader = DataLoader(dataset, batch_size=batch_size, sampler = test_sampler)

    return train_data_loader, test_data_loader

def plot_image_and_mask(dataloader):
    for img, mask in dataloader:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        img = img.permute(0, 2, 3, 1).numpy()
        mask = torch.argmax(mask, dim=1).numpy()

        ax[0].imshow(img[0])
        ax[1].imshow(mask[0])
        plt.show()
        break



