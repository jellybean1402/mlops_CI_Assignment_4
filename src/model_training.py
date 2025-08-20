import os
import yaml
import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

# Define the CNN Model
class SimpleCNN(nn.Module):
    def __init__(self, config):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, config['conv1_out_channels'], kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(config['conv1_out_channels'], config['conv2_out_channels'], kernel_size=3, padding=1)
        self.dropout = nn.Dropout(config['dropout_rate'])
        
        # Calculate the input size for the fully connected layer
        # This depends on the image size and the conv/pool layers
        fc_input_size = config['conv2_out_channels'] * (config['image_size'] // 4) * (config['image_size'] // 4)
        
        self.fc1 = nn.Linear(fc_input_size, config['fc1_out_features'])
        self.fc2 = nn.Linear(config['fc1_out_features'], config['num_classes'])
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def model_training(config_path):
    """
    Trains the CNN model based on the provided configuration.
    Args:
        config_path (str): Path to the params.yaml configuration file.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load data
    processed_data_path = config['processed_data_path']
    with open(os.path.join(processed_data_path, "train_loader.pkl"), "rb") as f:
        train_loader = pickle.load(f)
    with open(os.path.join(processed_data_path, "val_loader.pkl"), "rb") as f:
        val_loader = pickle.load(f)

    # Initialize model, loss, and optimizer
    model = SimpleCNN(config)
    criterion = nn.CrossEntropyLoss()
    
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'])
    else:
        raise ValueError(f"Optimizer {config['optimizer']} not supported.")

    # Training loop
    epochs = config['epochs']
    print("Starting model training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {running_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}')

    # Save the model
    model_path = config['model_path']
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), model_path)
    print(f"Model training complete. Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    model_training(config_path=args.config)
