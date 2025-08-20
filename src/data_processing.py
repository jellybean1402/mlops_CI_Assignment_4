import os
import yaml
import argparse
import torch
import pickle
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

def data_processing(config_path):
    """
    Processes raw data, applies transformations, splits the dataset,
    and saves DataLoaders.
    Args:
        config_path (str): Path to the params.yaml configuration file.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load params
    raw_data_path = os.path.join(config['data_path'], "raw")
    processed_data_path = config['processed_data_path']
    image_size = config['image_size']
    batch_size = config['batch_size']
    test_split = config['test_split_ratio']
    val_split = config['validation_split_ratio']
    
    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path)
        print(f"Created directory: {processed_data_path}")

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load dataset
    full_train_dataset = datasets.CIFAR10(root=raw_data_path, train=True, download=False, transform=transform)
    test_dataset = datasets.CIFAR10(root=raw_data_path, train=False, download=False, transform=transform)

    # Create train/validation split
    train_indices, val_indices = train_test_split(
        list(range(len(full_train_dataset))),
        test_size=val_split,
        stratify=full_train_dataset.targets
    )
    
    train_subset = Subset(full_train_dataset, train_indices)
    val_subset = Subset(full_train_dataset, val_indices)

    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Save DataLoaders
    with open(os.path.join(processed_data_path, "train_loader.pkl"), "wb") as f:
        pickle.dump(train_loader, f)
    with open(os.path.join(processed_data_path, "val_loader.pkl"), "wb") as f:
        pickle.dump(val_loader, f)
    with open(os.path.join(processed_data_path, "test_loader.pkl"), "wb") as f:
        pickle.dump(test_loader, f)
        
    print("Data processing complete. DataLoaders saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    data_processing(config_path=args.config)
