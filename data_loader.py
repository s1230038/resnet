import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import datasets
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder
from torch.utils.data.sampler import SubsetRandomSampler


def get_data_loaders(data_dir,
                     batch_size,
                     train_transform,
                     test_transform,
                     shuffle=True,
                     num_workers=4,
                     pin_memory=False):
    """
    Adapted from: https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
    
    Utility function for loading and returning train and test
    multi-process iterators over the CIFAR-10 dataset.
    If using CUDA, set pin_memory to True.
    
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - train_transform: pytorch transforms for the training set
    - test_transform: pytorch transofrms for the test set
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    
    Returns
    -------
    - train_loader: training set iterator.
    - test_loader:  test set iterator.
    """
    
    # Load the datasets
    '''
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=train_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False,
        download=True, transform=test_transform,
    )
    '''
    full_dataset_forTrain =ImageFolder(root=data_dir, transform=train_transform)
    full_dataset_forTest =ImageFolder(root=data_dir, transform=test_transform)
    
    train_size = int(0.8 * len(full_dataset_forTrain))
    test_size = len(full_dataset_forTrain) - train_size
    torch.manual_seed(torch.initial_seed())
    train_dataset, _ = torch.utils.data.random_split(
       full_dataset_forTrain, [train_size, test_size]
    )
    torch.manual_seed(torch.initial_seed())
    _, test_dataset = torch.utils.data.random_split(
       full_dataset_forTest, [train_size, test_size]
    )
    print(data_dir)
    print("full_data: "  + str(len(full_dataset_forTrain)) )
    print("train_dataset: " + str(len(train_dataset)) )
    print("test_dataset:"   + str(len(test_dataset)) )

    
    # Create loader objects
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory
    )
          
    return (train_loader, test_loader)


def plot_images(images, cls_true, cls_pred=None):
    """
    Plot images with labels.
    Adapted from https://github.com/Hvass-Labs/TensorFlow-Tutorials/
    """
    
    # CIFAR10 labels
    '''
    label_names = [
        'airplane',
        'automobile',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck'
    ]
    '''
    label_names = [
        'U3042',
        'U3044',
        'U3046',
        'U3048',
        'U304A',
        'U304B',
        'U304C',
        'U304D',
        'U304E',
        'U304F',
        'U3050',
        'U3051',
        'U3052',
        'U3053',
        'U3054',
        'U3055',
        'U3056',
        'U3057',
        'U3058',
        'U3059',
        'U305A',
        'U305B',
        'U305C',
        'U305D',
        'U305E',
        'U305F',
        'U3060',
        'U3061',
        'U3062',
        'U3064',
        'U3065',
        'U3066',
        'U3067',
        'U3068',
        'U3069',
        'U306A',
        'U306B',
        'U306C',
        'U306D',
        'U306E',
        'U306F',
        'U3070',
        'U3071',
        'U3072',
        'U3073',
        'U3074',
        'U3075',
        'U3076',
        'U3077',
        'U3078',
        'U3079',
        'U307A',
        'U307B',
        'U307C',
        'U307D',
        'U307E',
        'U307F',
        'U3080',
        'U3081',
        'U3082',
        'U3084',
        'U3086',
        'U3088',
        'U3089',
        'U308A',
        'U308B',
        'U308C',
        'U308D',
        'U308F',
        'U3090',
        'U3091',
        'U3092',
        'U3093'
    ]
    
    fig, axes = plt.subplots(3, 3)

    for i, ax in enumerate(axes.flat):
        # plot img
        ax.imshow(images[i, :, :, :], interpolation='spline16')

        # show true & predicted classes
        cls_true_name = label_names[cls_true[i]]
        if cls_pred is None:
            xlabel = "{0} ({1})".format(cls_true_name, cls_true[i])
        else:
            cls_pred_name = label_names[cls_pred[i]]
            xlabel = "True: {0}\nPred: {1}".format(
                cls_true_name, cls_pred_name
            )
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


