import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from efficient_kan import KAN
from efficient_dropkan import DropKAN
import matplotlib.pyplot as plt
import logging
import time
import os
import json
import gzip
from PIL import Image
from tqdm import tqdm

# Set up logging


def setup_logging(log_file='model_comparison.log'):
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

# Load Fashion-MNIST dataset


def load_fashion_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    class LocalFashionMNIST(Dataset):
        def __init__(self, root, train=True, transform=None):
            self.root = root
            self.transform = transform
            self.train = train

            if self.train:
                data_file = 'train-images-idx3-ubyte.gz'
                labels_file = 'train-labels-idx1-ubyte.gz'
            else:
                data_file = 't10k-images-idx3-ubyte.gz'
                labels_file = 't10k-labels-idx1-ubyte.gz'

            self.data = self._read_images(os.path.join(
                self.root, 'FashionMNIST', 'raw', data_file))
            self.targets = self._read_labels(os.path.join(
                self.root, 'FashionMNIST', 'raw', labels_file))

        def _read_images(self, path):
            with gzip.open(path, 'rb') as f:
                data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
            return data.reshape(-1, 28, 28)

        def _read_labels(self, path):
            with gzip.open(path, 'rb') as f:
                data = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
            return data

        def __getitem__(self, index):
            img, target = self.data[index], int(self.targets[index])
            img = Image.fromarray(img, mode='L')

            if self.transform is not None:
                img = self.transform(img)

            return img, target

        def __len__(self):
            return len(self.data)

    train_dataset = LocalFashionMNIST(
        root='./data', train=True, transform=transform)
    test_dataset = LocalFashionMNIST(
        root='./data', train=False, transform=transform)

    # Convert to binary classification (0-4: 0, 5-9: 1)
    train_dataset.targets = torch.tensor(
        train_dataset.targets >= 5, dtype=torch.float)
    test_dataset.targets = torch.tensor(
        test_dataset.targets >= 5, dtype=torch.float)

    logger.info(
        f"Loaded Fashion-MNIST dataset: {len(train_dataset)} training samples, {len(test_dataset)} test samples")
    return train_dataset, test_dataset

# Train function


def train_model(model, train_loader, criterion, optimizer, device, epochs=50):
    model.train()
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit='batch') as pbar:
            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X.view(batch_X.size(0), -1))
                loss = criterion(outputs.squeeze(), batch_y.float())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

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
        with tqdm(total=len(test_loader), desc="Evaluating", unit='batch') as pbar:
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X.view(batch_X.size(0), -1))
                y_true.extend(batch_y.cpu().numpy())
                y_pred.extend((outputs.squeeze() > 0).float().cpu().numpy())
                pbar.update(1)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return accuracy, precision, recall, f1

# Main comparison function


def compare_models(train_dataset, test_dataset, hidden_dims=[128, 64], batch_size=64, learning_rate=0.001, epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Define models
    input_dim = 28 * 28  # Fashion-MNIST image size
    output_dim = 1
    model_configs = [
        ("KAN", KAN, {}),
        ("DropKAN (dropout=0.1)", DropKAN, {
         "drop_rate": 0.1, "drop_mode": "dropout"}),
        ("DropKAN (postspline=0.1)", DropKAN, {
         "drop_rate": 0.1, "drop_mode": "postspline"}),
        ("DropKAN (postact=0.1)", DropKAN, {
         "drop_rate": 0.1, "drop_mode": "postact"}),
        ("DropKAN (postact=0.3)", DropKAN, {
         "drop_rate": 0.3, "drop_mode": "postact"}),
    ]

    results = {}

    for name, model_class, model_kwargs in model_configs:
        logger.info(f"\nTraining {name}")
        model = model_class([input_dim] + hidden_dims +
                            [output_dim], **model_kwargs).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        start_time = time.time()
        losses = train_model(model, train_loader, criterion,
                             optimizer, device, epochs)
        training_time = time.time() - start_time

        accuracy, precision, recall, f1 = evaluate_model(
            model, test_loader, device)

        results[name] = {
            "losses": losses,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "training_time": training_time
        }

        logger.info(
            f"{name} - Training completed in {training_time:.2f} seconds")
        logger.info(f"{name} - Test Accuracy: {accuracy:.4f}")
        logger.info(f"{name} - Test Precision: {precision:.4f}")
        logger.info(f"{name} - Test Recall: {recall:.4f}")
        logger.info(f"{name} - Test F1 Score: {f1:.4f}")

    return results


# Run the comparison
train_dataset, test_dataset = load_fashion_mnist()
results = compare_models(train_dataset, test_dataset)

# Log detailed results
logger.info("\nDetailed Comparison Results:")
for name, result in results.items():
    logger.info(f"\n{name}:")
    logger.info(f"  Final Test Accuracy: {result['accuracy']:.4f}")
    logger.info(f"  Final Test Precision: {result['precision']:.4f}")
    logger.info(f"  Final Test Recall: {result['recall']:.4f}")
    logger.info(f"  Final Test F1 Score: {result['f1']:.4f}")
    logger.info(f"  Training Time: {result['training_time']:.2f} seconds")
    logger.info(f"  Final Training Loss: {result['losses'][-1]:.4f}")

# Save results to a JSON file
json_file = 'tests/results/MNIST_KAN_model_comparison_results.json'
json_results = {name: {
    'accuracy': float(result['accuracy']),
    'precision': float(result['precision']),
    'recall': float(result['recall']),
    'f1_score': float(result['f1']),
    'training_time': float(result['training_time']),
    'final_loss': float(result['losses'][-1]),
    'loss_history': [float(loss) for loss in result['losses']]
} for name, result in results.items()}

with open(json_file, 'w') as file:
    json.dump(json_results, file, indent=4)

logger.info(f"Detailed results saved to '{json_file}'")