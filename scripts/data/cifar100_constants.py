"""Compute the CIFAR100 normalizing constants."""

import torch
from datasets import load_dataset

dataset = load_dataset("uoft-cs/cifar100", split="train").with_format("torch")
n_samples = len(dataset)
img_tensor = torch.stack(list(dataset["img"])).float() / 255.0
mean = torch.mean(img_tensor, dim=(0, 2, 3))
std = torch.std(img_tensor, dim=(0, 2, 3))
print(mean, std)
