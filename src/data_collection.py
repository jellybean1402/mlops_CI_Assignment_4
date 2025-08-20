import os
import yaml
import argparse
import torchvision.datasets as datasets

def data_collection(config_path):
    """
    Downloads the dataset specified in the params.yaml file.
    Args:
        config_path (str): Path to the params.yaml configuration file.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    dataset_name = config['dataset_name']
    data_path = config['data_path']
    raw_data_path = os.path.join(data_path, "raw")

    if not os.path.exists(raw_data_path):
        os.makedirs(raw_data_path)
        print(f"Created directory: {raw_data_path}")

    if dataset_name == "CIFAR10":
        print(f"Downloading {dataset_name} dataset to {raw_data_path}...")
        datasets.CIFAR10(root=raw_data_path, train=True, download=True)
        datasets.CIFAR10(root=raw_data_path, train=False, download=True)
        print("Download complete.")
    else:
        print(f"Dataset {dataset_name} is not supported.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    data_collection(config_path=args.config)
