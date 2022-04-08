# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from domainbed.lib.misc import DictDataset
import os, pickle
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate
import matplotlib.pyplot as plt
import pandas as pd

# -*- coding: utf-8 -*-

from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms.transforms import RandomHorizontalFlip

# from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
# from wilds.datasets.fmow_dataset import FMoWDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    "Debug28",
    "Debug224",
    # Small images
    "ColoredMNIST",
    "ColoredMNISTClean",
    "CustomColoredMnist",
    "RotatedMNIST",
    # Big images
    "VLCS",
    "PACS",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet",
    "SVIRO",
    "CelebA_Blond",
    # WILDS datasets
    "WILDSCamelyon",
    "WILDSFMoW",
    "ColoredMNISTLFF",
    "BAR",
    "Collage",
    "TwoDirections2D",
    "Spirals",
]


def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 5001  # Default, subclasses may override
    CHECKPOINT_FREQ = 300  # Default, subclasses may override
    N_WORKERS = 8  # Default, subclasses may override
    ENVIRONMENTS = None  # Subclasses should override
    INPUT_SHAPE = None  # Subclasses should override
    CUSTOM_DATASET = False  # may override. this means that the output format is different than (input, label)
    TEST_ENVS = [0]
    CLASSES = None
    NO_EVAL = None

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE), torch.randint(0, self.num_classes, (16,))
                )
            )


class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']


class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ['0', '1', '2']


class TwoDirections2D(MultipleDomainDataset):
    """
    Two Environments
    """
    N_STEPS = 5000
    CHECKPOINT_FREQ = 1000
    ENVIRONMENTS = ["train"]
    TEST_ENVS = [1,]
    INPUT_SHAPE = (2,)      # Subclasses should override

    DIRECTIONS = [
        [(1, 0), (-1, 0)],
        [(0, 1), (0, -1)],
        # [(-1, 0), (1, 0)],
        # [(1, 1), (-1, -1)],
        # [(-1, -1), (1, 1)],
    ]

    def after_training(self, model, dir, device):
        # plot the predictions on the whole space
        # what if multiple predictions ?
        r = torch.range(-2, 2, step=0.1)
        # print("r.shape", r.shape)
        points_list = torch.cartesian_prod(r, r).to(device)  # (20*20, 2)
        preds_d = model.predict({"x": points_list})
        if type(preds_d) != dict:
            preds_d = {"preds": preds_d}

        for key in preds_d:
            logits = preds_d[key]
            preds = torch.softmax(logits, 1)[:, 0]  # (20*20, 2)
            # print(points_list.shape)
            # print(preds.shape)
            # breakpoint()
            points_im = points_list.reshape(41, 41, -1)
            preds_im = preds.reshape(41, 41)
            # plot image and save into file
            fig, ax = plt.subplots()
            c = ax.pcolormesh(points_im[:, :, 0].cpu().numpy(), points_im[:, :, 1].cpu().numpy(), preds_im.detach().cpu().numpy(), vmin=0, vmax=1)
            fig.colorbar(c, ax=ax)
            points = torch.stack([p[0] for p in self.datasets[0]][:1000]).numpy()
            label = torch.stack([p[1] for p in self.datasets[0]][:1000]).numpy()
            points2 = torch.stack([p[0] for p in self.datasets[0]][-1000:]).numpy()
            label2 = torch.stack([p[1] for p in self.datasets[0]][-1000:]).numpy()
            points = np.concatenate([points, points2])
            label = np.concatenate([label, label2])
            ax.contour(points_im[:, :, 0].cpu().numpy(), points_im[:, :, 1].cpu().numpy(), preds_im.detach().cpu().numpy(), levels=[0.5])
            # breakpoint()
            ax.scatter(points[:, 0], points[:, 1], c=label, cmap="jet")
            # ax.scatter(points[:, 0], points[:, 1], c=label)
            plt.savefig(os.path.join(dir, f'predictions-{key}.png'))

    def __init__(self, data_dir=None, test_envs=None, hparams=None) -> None:
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2

        self.datasets = []

        num_samples_per_class = 10000
        for dir in self.DIRECTIONS:
            train_x_positive = torch.tensor(dir[0])[ None, :] + 0.1 * torch.randn(num_samples_per_class, 2)
            train_x_negative = torch.tensor(dir[1])[None, :] + 0.1 * torch.randn(num_samples_per_class, 2)
            train_y = torch.cat((torch.ones(num_samples_per_class), torch.zeros(num_samples_per_class))).long()

            self.datasets.append(
                TensorDataset(torch.cat((train_x_positive, train_x_negative)), train_y))


class MultipleEnvironmentMNIST(MultipleDomainDataset):

    def __init__(self, root, environments, dataset_transform, input_shape, num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data, original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets, original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes


class ColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']
    TEST_ENVS = [2]

    def __init__(self, root, test_envs, hparams):
        super(ColoredMNIST,
              self).__init__(root, [0.1, 0.2, 0.9], self.color_dataset, (
                  2,
                  28,
                  28,
              ), 2)

        self.input_shape = (
            2,
            28,
            28,
        )
        self.num_classes = 2

    proba_flip = 0.25

    def color_dataset(self, images, labels, environment):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels, self.torch_bernoulli_(self.proba_flip, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels, self.torch_bernoulli_(environment, len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class ColoredMNISTClean(ColoredMNIST):
    proba_flip = 0.0


class CustomCMNISTDataset(Dataset):
    def __init__(self, dict):
        super().__init__()
        self.dict = dict

    def __getitem__(self, index):
        return {
            "x": self.dict["x"][index],
            "y": self.dict["y"][index],
            "biases": {
                "color": self.dict["biases"]["color"][index],
                "shape": self.dict["biases"]["shape"][index],
            }
        }
    def __len__(self):
        return len(self.dict["x"])


class CustomColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']
    TEST_ENVS = [2]
    CUSTOM_DATASET = True

    def __init__(self, root, test_envs, hparams):
        super(CustomColoredMNIST, self).__init__(root, [0.1, 0.2, 0.9], self.color_dataset, (2, 28, 28,), 2)
        self.input_shape = (2, 28, 28,)
        self.num_classes = 2
        self.hparams = hparams

    def color_dataset(self, images, labels, environment):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        true_labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(true_labels, self.torch_bernoulli_(0.25, len(true_labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels, self.torch_bernoulli_(environment, len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return CustomCMNISTDataset({"x": x, "y": y, "biases": {"shape": true_labels, "color": colors}})

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class CustomGrayColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']
    TEST_ENVS = [2]
    CUSTOM_DATASET = True

    def __init__(self, root, test_envs, hparams):
        super().__init__(root, [0.1, 0.2, 0.9], self.color_dataset, (2, 28, 28,), 2)

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2
        self.hparams = hparams

    def color_dataset(self, images, labels, environment):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        true_labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(true_labels, self.torch_bernoulli_(0.25, len(true_labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = torch.zeros_like(labels)  # self.torch_xor_(labels,
        #             self.torch_bernoulli_(environment,
        #                                  len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return CustomCMNISTDataset({"x": x, "y": y, "biases": {"shape": true_labels, "color": colors}})

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']

    def __init__(self, root, test_envs, hparams):
        super(RotatedMNIST,
              self).__init__(root, [0, 15, 30, 45, 60, 75], self.rotate_dataset, (
                  1,
                  28,
                  28,
              ), 10)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(
                lambda x: rotate(x, angle, fill=(0,), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)


class MultipleEnvironmentImageFolder(MultipleDomainDataset):

    def __init__(self, root, test_envs, augment, hparams):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

        augment_transform = transforms.Compose(
            [
                # transforms.Resize((224,224)),
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.datasets = []
        for i, environment in enumerate(environments):

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path, transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (
            3,
            224,
            224,
        )
        self.num_classes = len(self.datasets[-1].classes)


class VLCS(MultipleEnvironmentImageFolder):
    # CHECKPOINT_FREQ = 300
    CHECKPOINT_FREQ = 50  # Default, subclasses may override
    ENVIRONMENTS = ["C", "L", "S", "V"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class PACS(MultipleEnvironmentImageFolder):
    # CHECKPOINT_FREQ = 300
    CHECKPOINT_FREQ = 100
    ENVIRONMENTS = ["A", "C", "P", "S"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 1000
    ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "domain_net/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 100
    ENVIRONMENTS = ["A", "C", "P", "R"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "office_home/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class TerraIncognita(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 100
    ENVIRONMENTS = ["100", "38", "43", "46"]
    # ENVIRONMENTS = ["L100", "L38", "L43", "L46"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "terra_incognita/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class SVIRO(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = [
        "aclass", "escape", "hilux", "i3", "lexus", "tesla", "tiguan", "tucson", "x5", "zoe"
    ]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "sviro/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class WILDSEnvironment:

    def __init__(self, wilds_dataset, metadata_name, metadata_value, transform=None):
        self.name = metadata_name + "_" + str(metadata_value)

        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_array = wilds_dataset.metadata_array
        subset_indices = torch.where(metadata_array[:, metadata_index] == metadata_value)[0]

        self.dataset = wilds_dataset
        self.indices = subset_indices
        self.transform = transform

    def __getitem__(self, i):
        x = self.dataset.get_input(self.indices[i])
        if type(x).__name__ != "Image":
            x = Image.fromarray(x)

        y = self.dataset.y_array[self.indices[i]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)


class WILDSDataset(MultipleDomainDataset):
    INPUT_SHAPE = (3, 224, 224)

    def __init__(self, dataset, metadata_name, test_envs, augment, hparams):
        super().__init__()

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

        augment_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.datasets = []

        for i, metadata_value in enumerate(self.metadata_values(dataset, metadata_name)):
            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            env_dataset = WILDSEnvironment(dataset, metadata_name, metadata_value, env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (
            3,
            224,
            224,
        )
        self.num_classes = dataset.n_classes

    def metadata_values(self, wilds_dataset, metadata_name):
        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_vals = wilds_dataset.metadata_array[:, metadata_index]
        return sorted(list(set(metadata_vals.view(-1).tolist())))


class WILDSCamelyon(WILDSDataset):
    ENVIRONMENTS = ["hospital_0", "hospital_1", "hospital_2", "hospital_3", "hospital_4"]

    def __init__(self, root, test_envs, hparams):
        dataset = Camelyon17Dataset(root_dir=root)
        super().__init__(dataset, "hospital", test_envs, hparams['data_augmentation'], hparams)


class WILDSFMoW(WILDSDataset):
    ENVIRONMENTS = ["region_0", "region_1", "region_2", "region_3", "region_4", "region_5"]

    def __init__(self, root, test_envs, hparams):
        dataset = FMoWDataset(root_dir=root)
        super().__init__(dataset, "region", test_envs, hparams['data_augmentation'], hparams)


class ColoredMNISTLFFDataset(Dataset):
    def __init__(
        self,
        split="train",
        dir="data/lff/ColoredMNIST-Skewed0.01-Severity4",
        hparams=None,
    ):

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.dir = dir
        self.images = np.load(os.path.join(dir, split, "images.npy"))
        self.attrs = np.load(os.path.join(dir, split, "attrs.npy"))
        self.attrs_names = ["digit", "color"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        x = self.images[index]
        img = self.transform(x)
        img = img.flatten()
        number, color = self.attrs[index]
        out = {
            "x": img,
            "y": torch.tensor(number).long(),
            "index": index,
        }
        return out


class ColoredMNISTLFF(MultipleDomainDataset):
    ENVIRONMENTS = ["train", "test"]
    CUSTOM_DATASET = True
    TEST_ENVS = [1]
    INPUT_SHAPE = (3, 28, 28)
    N_STEPS = 20000
    CHECKPOINT_FREQ = 400

    def __init__(self, root, test_envs, hparams):
        train = ColoredMNISTLFFDataset(split="train", dir=root, hparams=hparams)
        test = ColoredMNISTLFFDataset(split="valid", dir=root, hparams=hparams)
        self.num_classes = 10
        self.datasets = [train, test]
        self.input_shape = (2352,)


class BARDataset(Dataset):
    """
    Download dataset from https://github.com/alinlab/BAR
    and put it in data/BAR
    """

    classes = ["climbing", "diving", "fishing", "pole vaulting", "racing", "throwing"]
    classes_to_id = {c: i for (i, c) in enumerate(classes)}

    def __init__(
        self,
        data_dir="data/BAR",
        split="train",
        train_transform=None,
        test_transform=None,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.split_dir = os.path.join(data_dir, split)
        filenames = os.listdir(self.split_dir)
        self.filenames = filenames

        if split == "train":
            if train_transform is None:
                self.transform = transforms.Compose(
                    [
                        transforms.RandomResizedCrop((224, 224)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ]
                )
            else:
                self.transform = train_transform

        elif split == "test":
            if test_transform is None:
                self.transform = transforms.Compose(
                    [
                        transforms.CenterCrop((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ]
                )
            else:
                self.transform = test_transform

    def __getitem__(self, index: int):
        filename = self.filenames[index]
        label = filename.split("_")[0]
        label_id = self.classes_to_id[label]
        im = Image.open(os.path.join(self.split_dir, filename)).convert("RGB")
        return {
            "x": self.transform(im),
            "y": label_id,
            "index": index,
        }

    def __len__(self):
        # int
        return len(self.filenames)


class BAR(MultipleDomainDataset):
    ENVIRONMENTS = ["train", "test"]
    CUSTOM_DATASET = True
    TEST_ENVS = [1]
    INPUT_SHAPE = None  # doesn't matter
    N_STEPS = 540
    CHECKPOINT_FREQ = 24  # 4 epochs
    CLASSES = ["climbing", "diving", "fishing", "vaulting", "racing", "throwing"]

    def __init__(self, root, test_envs, hparams):
        data_dir = os.path.join(root, "BAR")
        train = BARDataset(split="train", data_dir=data_dir)
        test = BARDataset(split="test", data_dir=data_dir)
        self.num_classes = 6
        self.datasets = [train, test]
        self.input_shape = self.INPUT_SHAPE


class Collage(MultipleDomainDataset):
    ENVIRONMENTS = ["train", "cifar", "fashion", "mnist", "svhn", "val-all"]
    CUSTOM_DATASET = False
    TEST_ENVS = [1, 2, 3, 4, 5]
    INPUT_SHAPE = (16, 16)  # doesn't matter
    N_STEPS = 13000   # 65 epochs
    CHECKPOINT_FREQ = 1000
    CLASSES = ["zero", "one"]
    NO_EVAL = ["env_0_in"]

    def __init__(
        self, root, test_envs, hparams,
        # data_dir="data/collages-4blocks-randomOrder0-downsampling2",
        # batch_size=256,
        # num_workers=4,
        # pin_memory=True,
    ):
        super().__init__()
        # self.data_dir = data_dir
        # self.batch_size = batch_size
        # self.num_workers = num_workers
        # self.pin_memory = pin_memory
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        data_dir = os.path.join(root, "collage", "collages-4blocks-randomOrder0-downsampling2")
        self.transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Grayscale(),
                torchvision.transforms.ToTensor(),
            ]
        )

        self.val_dataset_names = ["cifar", "fashion", "mnist", "svhn"]
        self.datasets = [
            torchvision.datasets.ImageFolder(
            root=os.path.join(data_dir, "train-all"), transform=self.transforms)
        ]
        for name in self.val_dataset_names:
            self.datasets.append(
                torchvision.datasets.ImageFolder(
                    root=os.path.join(data_dir, f"test-{name}"),
                    transform=self.transforms,
            )
            )
        self.datasets.append(
            torchvision.datasets.ImageFolder(
                root=os.path.join(data_dir, f"val-all"), transform=self.transforms,
            )
        )


# this class is adapted from https://github.com/chingyaoc/fair-mixup/blob/master/celeba/main_dp.py
class CelebA(torch.utils.data.Dataset):
    def __init__(self, dataframe, folder_dir, target_id, transform=None, cdiv=0, ccor=0):
        self.dataframe = dataframe
        self.folder_dir = folder_dir
        self.target_id = target_id
        self.transform = transform
        self.file_names = dataframe.index
        self.targets = np.concatenate(dataframe.labels.values).astype(int)
        gender_id = 20

        target_idx0 = np.where(self.targets[:, target_id] == 0)[0]
        target_idx1 = np.where(self.targets[:, target_id] == 1)[0]
        gender_idx0 = np.where(self.targets[:, gender_id] == 0)[0]
        gender_idx1 = np.where(self.targets[:, gender_id] == 1)[0]
        nontarget_males = list(set(gender_idx1) & set(target_idx0))
        nontarget_females = list(set(gender_idx0) & set(target_idx0))
        target_males = list(set(gender_idx1) & set(target_idx1))
        target_females = list(set(gender_idx0) & set(target_idx1))

        u1 = len(nontarget_males) - int((1 - ccor) * (len(nontarget_males) - len(nontarget_females)))
        u2 = len(target_females) - int((1 - ccor) * (len(target_females) - len(target_males)))
        selected_idx = nontarget_males[:u1] + nontarget_females + target_males + target_females[:u2]
        self.targets = self.targets[selected_idx]
        self.file_names = self.file_names[selected_idx]

        target_idx0 = np.where(self.targets[:, target_id] == 0)[0]
        target_idx1 = np.where(self.targets[:, target_id] == 1)[0]
        gender_idx0 = np.where(self.targets[:, gender_id] == 0)[0]
        gender_idx1 = np.where(self.targets[:, gender_id] == 1)[0]
        nontarget_males = list(set(gender_idx1) & set(target_idx0))
        nontarget_females = list(set(gender_idx0) & set(target_idx0))
        target_males = list(set(gender_idx1) & set(target_idx1))
        target_females = list(set(gender_idx0) & set(target_idx1))

        selected_idx = nontarget_males + nontarget_females[:int(len(nontarget_females) * (1 - cdiv))] + target_males + target_females[:int(len(target_females) * (1 - cdiv))]
        self.targets = self.targets[selected_idx]
        self.file_names = self.file_names[selected_idx]

        target_idx0 = np.where(self.targets[:, target_id] == 0)[0]
        target_idx1 = np.where(self.targets[:, target_id] == 1)[0]
        gender_idx0 = np.where(self.targets[:, gender_id] == 0)[0]
        gender_idx1 = np.where(self.targets[:, gender_id] == 1)[0]
        nontarget_males = list(set(gender_idx1) & set(target_idx0))
        nontarget_females = list(set(gender_idx0) & set(target_idx0))
        target_males = list(set(gender_idx1) & set(target_idx1))
        target_females = list(set(gender_idx0) & set(target_idx1))
        print(len(nontarget_males), len(nontarget_females), len(target_males), len(target_females))

        self.targets = self.targets[:, self.target_id]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.folder_dir, self.file_names[index]))
        label = self.targets[index]
        if self.transform:
            image = self.transform(image)
        return image, label


class CelebA_Blond(MultipleDomainDataset):
    ENVIRONMENTS = ["unbalanced_1", "unbalanced_2", "balanced"]
    N_STEPS = 2001
    CHECKPOINT_FREQ = 200

    def __init__(self, root, test_envs, hparams):
        super().__init__()
        environments = self.ENVIRONMENTS
        print(environments)

        self.input_shape = (3, 224, 224,)
        self.num_classes = 2 # blond or not

        dataframes = []
        for env_name in ('tr_env1', 'tr_env2', 'te_env'):
            with open(f'{root}/celeba/blond_split/{env_name}_df.pickle', 'rb') as handle:
                dataframes.append(pickle.load(handle))
        tr_env1, tr_env2, te_env = dataframes

        orig_w = 178
        orig_h = 218
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        images_path = f'{root}/celeba/img_align_celeba'
        transform = transforms.Compose([
            transforms.CenterCrop(min(orig_w, orig_h)),
            transforms.Resize(self.input_shape[1:]),
            transforms.ToTensor(),
            normalize,
        ])

        if hparams['data_augmentation']:
            augment_transform = transforms.Compose([
                transforms.RandomResizedCrop(self.input_shape[1:],
                                             scale=(0.7, 1.0), ratio=(1.0, 1.3333333333333333)),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.0),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])

            if hparams.get('test_data_augmentation', False):
                transform = augment_transform
        else:
            augment_transform = transform

        cdiv = hparams.get('cdiv', 0)
        ccor = hparams.get('ccor', 1)

        target_id = 9
        tr_dataset_1 = CelebA(pd.DataFrame(tr_env1), images_path, target_id, transform=augment_transform,
                              cdiv=cdiv, ccor=ccor)
        tr_dataset_2 = CelebA(pd.DataFrame(tr_env2), images_path, target_id, transform=augment_transform,
                              cdiv=cdiv, ccor=ccor)
        te_dataset = CelebA(pd.DataFrame(te_env), images_path, target_id, transform=transform)

        self.datasets = [tr_dataset_1, tr_dataset_2, te_dataset]


class Spirals(MultipleDomainDataset):
    CHECKPOINT_FREQ = 10
    N_WORKERS = 1  # Default, subclasses may override
    ENVIRONMENTS = [str(i) for i in range(16)]

    def __init__(self, root, test_env, hparams):
        super().__init__()
        self.datasets = []

        test_dataset = self.make_tensor_dataset(env='test')
        self.datasets.append(test_dataset)
        for env in self.ENVIRONMENTS:
            env_dataset = self.make_tensor_dataset(env=env, seed=int(env))
            self.datasets.append(env_dataset)

        self.input_shape = (18,)
        self.num_classes = 2

    def make_tensor_dataset(
        self,
        env,
        n_examples=1024,
        n_envs=16,
        n_revolutions=3,
        n_dims=16,
        flip_first_signature=False,
        seed=0
    ):

        if env == 'test':
            inputs, labels = self.generate_environment(
                2000,
                n_rotations=n_revolutions,
                env=env,
                n_envs=n_envs,
                n_dims_signatures=n_dims,
                seed=2**32 - 1
            )
        else:
            inputs, labels = self.generate_environment(
                n_examples,
                n_rotations=n_revolutions,
                env=env,
                n_envs=n_envs,
                n_dims_signatures=n_dims,
                seed=seed
            )
        if flip_first_signature:
            inputs[:1, 2:] = -inputs[:1, 2:]

        return TensorDataset(torch.tensor(inputs), torch.tensor(labels))

    def generate_environment(
        self, n_examples, n_rotations, env, n_envs, n_dims_signatures, seed=None
    ):
        """
        env must either be "test" or an int between 0 and n_envs-1
        n_dims_signatures: how many dimensions for the signatures (spirals are always 2)
        seed: seed for numpy
        """
        assert env == 'test' or 0 <= int(env) < n_envs

        # Generate fixed dictionary of signatures
        rng = np.random.RandomState(seed)

        signatures_matrix = rng.randn(n_envs, n_dims_signatures)

        radii = rng.uniform(0.08, 1, n_examples)
        angles = 2 * n_rotations * np.pi * radii

        labels = rng.randint(0, 2, n_examples)
        angles = angles + np.pi * labels

        radii += rng.uniform(-0.02, 0.02, n_examples)
        xs = np.cos(angles) * radii
        ys = np.sin(angles) * radii

        if env == 'test':
            signatures = rng.randn(n_examples, n_dims_signatures)
        else:
            env = int(env)
            signatures_labels = np.array(labels * 2 - 1).reshape(1, -1)
            signatures = signatures_matrix[env] * signatures_labels.T

        signatures = np.stack(signatures)
        mechanisms = np.stack((xs, ys), axis=1)
        mechanisms /= mechanisms.std(axis=0)  # make approx unit variance (signatures already are)
        inputs = np.hstack((mechanisms, signatures))

        return inputs.astype(np.float32), labels.astype(np.long)


# nico dataset
# https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Learning_Invariant_Representations_and_Risks_for_Semi-Supervised_Domain_Adaptation_CVPR_2021_paper.pdf
# https://github.com/Luodian/Learning-Invariant-Representations-and-Risks/blob/main/dset_loaders/nico.py
