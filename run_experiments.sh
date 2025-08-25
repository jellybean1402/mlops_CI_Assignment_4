#!/bin/bash

# This script queues and runs 20 DVC experiments with varying hyperparameters.

echo "Queuing experiments..."

# Experiment Set 1: Varying Learning Rate and Dropout (Optimizer: Adam)
dvc exp run --queue --set-param 'learning_rate=0.01' --set-param 'dropout_rate=0.2'
dvc exp run --queue --set-param 'learning_rate=0.01' --set-param 'dropout_rate=0.3'
dvc exp run --queue --set-param 'learning_rate=0.001' --set-param 'dropout_rate=0.2'
dvc exp run --queue --set-param 'learning_rate=0.001' --set-param 'dropout_rate=0.3'
dvc exp run --queue --set-param 'learning_rate=0.0005' --set-param 'dropout_rate=0.25'

# Experiment Set 2: Varying Epochs (Optimizer: Adam, lr=0.001)
dvc exp run --queue --set-param 'epochs=10'
dvc exp run --queue --set-param 'epochs=20'
dvc exp run --queue --set-param 'epochs=25'

# Experiment Set 3: Using SGD Optimizer
dvc exp run --queue --set-param 'optimizer=SGD' --set-param 'learning_rate=0.01'
dvc exp run --queue --set-param 'optimizer=SGD' --set-param 'learning_rate=0.005'

# Experiment Set 4: Varying Model Architecture (Optimizer: Adam, lr=0.001, epochs=15)
dvc exp run --queue --set-param 'conv1_out_channels=16' --set-param 'conv2_out_channels=32'
dvc exp run --queue --set-param 'conv1_out_channels=16' --set-param 'conv2_out_channels=64'
dvc exp run --queue --set-param 'conv1_out_channels=32' --set-param 'conv2_out_channels=128'
dvc exp run --queue --set-param 'fc1_out_features=64'
dvc exp run --queue --set-param 'fc1_out_features=256'

# Experiment Set 5: Combined Variations
dvc exp run --queue --set-param 'epochs=20' --set-param 'learning_rate=0.005' --set-param 'dropout_rate=0.3'
dvc exp run --queue --set-param 'epochs=20' --set-param 'optimizer=SGD' --set-param 'learning_rate=0.01'
dvc exp run --queue --set-param 'conv1_out_channels=16' --set-param 'fc1_out_features=64'
dvc exp run --queue --set-param 'learning_rate=0.01' --set-param 'epochs=10'
dvc exp run --queue --set-param 'dropout_rate=0.5' --set-param 'epochs=20'


echo "All 20 experiments have been queued."

# Run all queued experiments. Adjust --jobs to your machine's capability.
dvc exp run --run-all --jobs 2

echo "Experiment runs complete."

