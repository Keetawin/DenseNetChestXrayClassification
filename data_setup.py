"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""
import os

from torchvision import datasets, transforms
from torch.utils.data import random_split,DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS
):
  """Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                             test_dir=path/to/test_dir,
                             transform=some_transform,
                             batch_size=32,
                             num_workers=4)
  """
  # Use ImageFolder to create dataset(s)
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  # Get class names
  class_names = train_data.classes

  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names



def split_train_validation(train_dataloader, validation_split=0.1):
    """
    Splits the provided training DataLoader into new training and validation DataLoaders.

    Args:
        train_dataloader: The original training DataLoader to be split.
        validation_split: The percentage of data to be used for validation.

    Returns:
        A tuple of (new_train_dataloader, validation_dataloader).
    """
    # Extract the dataset from the original DataLoader
    original_dataset = train_dataloader.dataset

    # Calculate sizes of new training and validation sets
    num_train = len(original_dataset)
    split_size = int(validation_split * num_train)
    train_size = num_train - split_size

    # Split the dataset
    new_train_dataset, val_dataset = random_split(original_dataset, [train_size, split_size])

    # Create new DataLoaders
    new_train_dataloader = DataLoader(
        new_train_dataset,
        batch_size=train_dataloader.batch_size,
        shuffle=True,
        num_workers=train_dataloader.num_workers,
        pin_memory=train_dataloader.pin_memory,
    )
    validation_dataloader = DataLoader(
        val_dataset,
        batch_size=train_dataloader.batch_size,
        shuffle=False,
        num_workers=train_dataloader.num_workers,
        pin_memory=train_dataloader.pin_memory,
    )

    return new_train_dataloader, validation_dataloader