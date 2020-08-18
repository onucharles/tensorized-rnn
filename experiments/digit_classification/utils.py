import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split


def data_generator(root, batch_size, train_frac=5.0/6):
    if train_frac == 1.0:
        raise ValueError("Training set fraction cannot be 1. You need a validation set.")

    train_data = datasets.MNIST(root=root, train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]))
    len_train_set = int(train_frac * len(train_data))
    train_set, val_set = random_split(train_data, [len_train_set, len(train_data) - len_train_set])

    test_set = datasets.MNIST(root=root, train=False, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ]))
    print(f"train: {len(train_set)}\tval: {len(val_set)}\ttest: {len(test_set)}")

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                        num_workers=12)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                        num_workers=12)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                        num_workers=12)
    return train_loader, val_loader, test_loader


def count_model_params(model):
    """
    Returns number of trainable and non-trainable parameters in a model.
    :param model: A PyTorch nn.Module object.
    :return: A tuple (train_params_count, non_train_params_count)
    """

    train_params_count = 0
    non_train_params_count = 0
    for p in model.parameters():
        if p.requires_grad:
            train_params_count += p.numel()
        else:
            non_train_params_count += p.numel()

    return train_params_count, non_train_params_count
