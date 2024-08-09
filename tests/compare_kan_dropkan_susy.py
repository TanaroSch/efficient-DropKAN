import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import time
import json
from tqdm import tqdm
from efficient_kan import KAN
from efficient_dropkan import DropKAN
import os
import multiprocessing
from functools import partial

# Set up logging
def setup_logging(log_file='comprehensive_model_comparison.log'):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler()
                        ])
    return logging.getLogger(__name__)


logger = setup_logging()

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load SUSY dataset
def load_susy_dataset(file_path='data/SUSY.csv', sample_size=None):
    logger.info("Loading SUSY dataset...")
    data = pd.read_csv(file_path, header=None)

    if sample_size:
        data = data.sample(n=sample_size, random_state=42)

    X = data.iloc[:, 1:].values  # Features
    y = data.iloc[:, 0].values   # Labels

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    logger.info(
        f"Loaded SUSY dataset: {len(train_dataset)} training samples, {len(test_dataset)} test samples")
    return train_dataset, test_dataset

# Train function
def train_model(model, train_loader, criterion, optimizer, device, epochs=50):
    model.train()
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch_X, batch_y in progress_bar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        logger.info(
            f"Epoch {epoch+1}/{epochs} completed, Average Loss: {avg_loss:.4f}")

    return losses

# Evaluate function
def evaluate_model(model, test_loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating", leave=False)
        for batch_X, batch_y in progress_bar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            y_true.extend(batch_y.cpu().numpy())
            y_pred.extend((outputs.squeeze() > 0.5).float().cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return accuracy, precision, recall, f1

# Function to train and evaluate a single model configuration
def train_and_evaluate_model(model_config, train_dataset, test_dataset, batch_size, learning_rate, epochs, results_file):
    name, model_class, model_kwargs, hidden_dims = model_config

    # Check if model has already been trained
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            existing_results = json.load(f)
        if name in existing_results:
            logger.info(f"Skipping {name} as it has already been trained.")
            return None

    device = torch.device("cpu")

    logger.info(f"training {name}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = model_class([18] + hidden_dims + [1], **model_kwargs).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    start_time = time.time()
    losses = train_model(model, train_loader, criterion,
                         optimizer, device, epochs)
    training_time = time.time() - start_time

    accuracy, precision, recall, f1 = evaluate_model(
        model, test_loader, device)

    return {
        "name": name,
        "loss_history": [float(loss) for loss in losses],
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "training_time": float(training_time),
        "params": int(sum(p.numel() for p in model.parameters()))
    }

# Main comparison function
def compare_models(train_dataset, test_dataset, batch_size=256, learning_rate=0.001, epochs=50):
    # Define different architectures
    architectures = {
        "Original": [128, 64],
        "Deep": [256, 256, 128, 128, 64],
        "Wide": [256, 256, 256, 128, 128],
        "Pyramid": [512, 256, 128, 64, 32],
        "Bottleneck": [256, 128, 64, 128, 256]
    }

    # Define model configurations
    model_configs = [
        ("KAN", KAN, {}),
        ("DropKAN (dropout=0.1)", DropKAN, {
         "drop_rate": 0.1, "drop_mode": "dropout"}),
        ("DropKAN (postspline=0.1)", DropKAN, {
         "drop_rate": 0.1, "drop_mode": "postspline"}),
        ("DropKAN (postact=0.1)", DropKAN, {
         "drop_rate": 0.1, "drop_mode": "postact"}),
        ("DropKAN (postact=0.2)", DropKAN, {
         "drop_rate": 0.2, "drop_mode": "postact"}),
        ("DropKAN (postact=0.3)", DropKAN, {
         "drop_rate": 0.3, "drop_mode": "postact"}),
    ]

    # Create a list of all model configurations
    all_configs = []
    for arch_name, hidden_dims in architectures.items():
        for name, model_class, model_kwargs in model_configs:
            model_name = f"{name}_{arch_name}"
            all_configs.append(
                (model_name, model_class, model_kwargs, hidden_dims))

    # Use CPU multithreading
    num_cores = max(1, multiprocessing.cpu_count() - 2)
    logger.info(f"Using {num_cores} CPU cores for parallel processing")

    results_file = 'SUSY_comprehensive_model_comparison_results.json'

    # Create a partial function with fixed arguments
    train_and_evaluate_partial = partial(
        train_and_evaluate_model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        results_file=results_file
    )

    # Use multiprocessing to train and evaluate models in parallel
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.map(train_and_evaluate_partial, all_configs)

    # Filter out skipped models and convert to dictionary
    results_dict = {result['name']
        : result for result in results if result is not None}

    return results_dict


if __name__ == '__main__':
    # required for Windows compatibility
    multiprocessing.freeze_support()

    # Run the comparison
    train_dataset, test_dataset = load_susy_dataset(
        sample_size=500000)  # Adjust sample size as needed
    results = compare_models(train_dataset, test_dataset)

    # Save results to a JSON file
    json_file = 'tests/results/SUSY_comprehensive_model_comparison_results.json'

    # Load existing results if file exists
    if os.path.exists(json_file):
        with open(json_file, 'r') as file:
            existing_results = json.load(file)
    else:
        existing_results = {}

    # Update existing results with new results
    existing_results.update(results)

    # Save updated results
    with open(json_file, 'w') as file:
        json.dump(existing_results, file, indent=4)

    logger.info(
        f"Detailed comprehensive comparison results saved to '{json_file}'")

    # Create a summary table
    summary_data = []
    for name, result in existing_results.items():
        summary_data.append([
            name,
            result['accuracy'],
            result['precision'],
            result['recall'],
            result['f1'],
            result['training_time'],
            result['loss_history'][-1],
            result['params']
        ])

    summary_df = pd.DataFrame(summary_data, columns=[
                              'Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Training Time', 'Final Loss', 'Num Parameters'])
    summary_df = summary_df.sort_values('F1 Score', ascending=False)
    summary_df.to_csv('tests/results/model_comparison_summary.csv', index=False)
    logger.info("Summary table saved as 'model_comparison_summary.csv'")
