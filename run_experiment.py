# run_experiment.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor
from torch.utils.data import DataLoader
from torchprofile import profile_macs
import numpy as np
from collections import defaultdict, OrderedDict
from tqdm import tqdm
import copy

# ======== Model definition (VGG) ========
class VGG(nn.Module):
    ARCH = [64, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

    def __init__(self):
        super().__init__()
        layers = []
        counts = defaultdict(int)

        def add(name, layer):
            layers.append((f"{name}{counts[name]}", layer))
            counts[name] += 1

        in_channels = 3
        for x in self.ARCH:
            if x != 'M':
                add("conv", nn.Conv2d(in_channels, x, 3, padding=1, bias=False))
                add("bn", nn.BatchNorm2d(x))
                add("relu", nn.ReLU(True))
                in_channels = x
            else:
                add("pool", nn.MaxPool2d(2))

        self.backbone = nn.Sequential(OrderedDict(layers))
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        x = self.backbone(x)
        x = x.mean([2, 3])  # global avgpool
        x = self.classifier(x)
        return x


# ======== Evaluation ========
@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    total, correct = 0, 0
    for x, y in tqdm(dataloader, desc="Evaluating", leave=False):
        x, y = x.cuda(), y.cuda()
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


# ======== Pruning ========
def fine_grained_prune(tensor, sparsity):
    num_zeros = round(tensor.numel() * sparsity)
    importance = tensor.abs().flatten()
    threshold = torch.kthvalue(importance, num_zeros).values
    mask = torch.gt(tensor.abs(), threshold)
    tensor.mul_(mask)
    return mask


class FineGrainedPruner:
    def __init__(self, model, sparsity_dict):
        self.masks = self.prune(model, sparsity_dict)

    @staticmethod
    @torch.no_grad()
    def prune(model, sparsity_dict):
        masks = dict()
        for name, param in model.named_parameters():
            if param.dim() > 1:
                masks[name] = fine_grained_prune(param, sparsity_dict[name])
        return masks

    @torch.no_grad()
    def apply(self, model):
        for name, param in model.named_parameters():
            if name in self.masks:
                param.mul_(self.masks[name])


# ======== Utilities ========
def get_model_size(model, count_nonzero_only=False):
    total = 0
    for param in model.parameters():
        total += param.count_nonzero() if count_nonzero_only else param.numel()
    return total * 32  # bits


# ======== Main ========
def main():
    # Data
    image_size = 32
    transform = {
        "train": Compose([RandomCrop(image_size, padding=4), RandomHorizontalFlip(), ToTensor()]),
        "test": ToTensor()
    }
    dataset = {
        split: CIFAR10(root="data", train=(split == "train"), download=True, transform=transform[split])
        for split in ["train", "test"]
    }
    dataloader = {
        split: DataLoader(dataset[split], batch_size=512, shuffle=(split == "train"), num_workers=2, pin_memory=True)
        for split in ["train", "test"]
    }

    # Model setup
    model = VGG().cuda()
    checkpoint = torch.load("vgg.cifar.pretrained.pth", map_location="cpu")
    model.load_state_dict(checkpoint['state_dict'])
    base_model = copy.deepcopy(model)

    # Evaluate dense model
    dense_acc = evaluate(model, dataloader["test"])
    dense_size = get_model_size(model)
    print(f"[Dense] Accuracy: {dense_acc:.2f}%, Model Size: {dense_size / (1024*1024):.2f} MiB")

    # Sparsity configs
    sparsity_levels = [0.4, 0.5, 0.6]
    for sparsity in sparsity_levels:
        model = copy.deepcopy(base_model)
        sparsity_dict = {
            name: sparsity for name, param in model.named_parameters() if param.dim() > 1
        }
        pruner = FineGrainedPruner(model, sparsity_dict)
        pruner.apply(model)
        acc = evaluate(model, dataloader["test"])
        size = get_model_size(model, count_nonzero_only=True)
        print(f"[Sparsity {sparsity:.1f}] Accuracy: {acc:.2f}%, Size: {size / (1024*1024):.2f} MiB")

if __name__ == "__main__":
    main()
