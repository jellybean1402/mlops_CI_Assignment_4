import os
import yaml
import argparse
import pickle
import json
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from model_training import SimpleCNN # Import model class from training script

def model_evaluation(config_path):
    """
    Evaluates the trained model on the test dataset and saves metrics.
    Args:
        config_path (str): Path to the params.yaml configuration file.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load test data
    processed_data_path = config['processed_data_path']
    with open(os.path.join(processed_data_path, "test_loader.pkl"), "rb") as f:
        test_loader = pickle.load(f)

    # Load model
    model = SimpleCNN(config)
    model.load_state_dict(torch.load(config['model_path']))
    model.eval()

    all_preds = []
    all_labels = []

    print("Evaluating model on the test set...")
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    # Save metrics
    reports_path = config['reports_path']
    reports_dir = os.path.dirname(reports_path)
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)

    with open(reports_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print("Evaluation complete.")
    print(f"Metrics: {metrics}")
    print(f"Metrics saved to {reports_path}")
    
    # Generate and save classification report
    class_report = classification_report(all_labels, all_preds)
    report_path = os.path.join(reports_dir, "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write(class_report)
    print(f"Classification report saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    model_evaluation(config_path=args.config)
