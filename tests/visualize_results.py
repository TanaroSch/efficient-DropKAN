import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Create results directory if it doesn't exist
results_dir = "tests/results/"
os.makedirs(results_dir, exist_ok=True)

# File paths (replace these with actual paths when using)
mnist_file_path = f"{results_dir}MNIST_KAN_model_comparison_results.json"
susy_file_path = f"{results_dir}SUSY_comprehensive_model_comparison_results.json"


# Load MNIST data
with open(mnist_file_path, 'r') as f:
    mnist_data = json.load(f)

# Load SUSY data
with open(susy_file_path, 'r') as f:
    susy_data = json.load(f)

# Define a more visually appealing color palette
# Using seaborn's husl palette for distinct, appealing colors
color_palette = sns.color_palette("husl", 8)

# Function to assign colors and groups


def assign_color_and_group(model_name):
    if model_name.startswith('KAN'):
        return color_palette[0]
    elif 'dropout=' in model_name:
        return color_palette[1]
    elif 'postspline=' in model_name:
        return color_palette[2]
    elif 'postact=' in model_name:
        try:
            value = float(model_name.split('postact=')[1].split(')')[0])
            if value <= 0.1:
                return color_palette[3]
            elif value <= 0.2:
                return color_palette[4]
            else:
                return color_palette[5]
        except ValueError:
            return color_palette[-1]  # Default color if parsing fails
    return color_palette[-1]  # Default color


def get_architecture(model_name):
    if '_Original' in model_name:
        return 'Original'
    elif '_Deep' in model_name:
        return 'Deep'
    elif '_Wide' in model_name:
        return 'Wide'
    elif '_Pyramid' in model_name:
        return 'Pyramid'
    elif '_Bottleneck' in model_name:
        return 'Bottleneck'
    else:
        return 'Other'

# Function to create performance metric comparison plot


def plot_performance_metrics(data, title):
    metrics = ['accuracy', 'precision', 'recall']
    if 'f1_score' in next(iter(data.values())):
        metrics.append('f1_score')
    elif 'f1' in next(iter(data.values())):
        metrics.append('f1')

    fig, axes = plt.subplots(len(metrics), 1, figsize=(15, 7*len(metrics)))
    fig.suptitle(f"{title} - Performance Metrics Comparison",
                 y=0.98)  # Adjusted y position

    models = list(data.keys())
    colors = [assign_color_and_group(model) for model in models]
    architectures = [get_architecture(model) for model in models]

    x = np.arange(len(models))
    width = 0.8

    for i, metric in enumerate(metrics):
        ax = axes[i] if len(metrics) > 1 else axes
        values = [data[model].get(metric, 0) for model in models]

        bars = ax.bar(x, values, width, color=colors)

        # Add value labels inside the bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height*0.95,
                    f'{height:.4f}',
                    ha='center', va='top', rotation=45, fontsize=8, color='white', fontweight='bold')

        ax.set_title(metric.capitalize())
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')

        # Add space between different architectures
        prev_arch = None
        for j, arch in enumerate(architectures):
            if prev_arch and arch != prev_arch:
                ax.axvline(j - 0.5, color='black', linestyle='--', alpha=0.3)
            prev_arch = arch

        ax.set_xlabel('Models')
        ax.set_ylabel(metric.capitalize())

    plt.tight_layout()
    # Adjusted top margin and increased hspace
    plt.subplots_adjust(top=0.93, bottom=0.2, hspace=0.6)
    plt.savefig(f"{results_dir}{title.lower()}_performance_metrics.png",
                dpi=300, bbox_inches='tight')
    plt.close()

# Function to plot loss history


def plot_loss_history(data, title):
    plt.figure(figsize=(15, 8))
    for model, values in data.items():
        if 'loss_history' in values:
            color = assign_color_and_group(model)
            plt.plot(values['loss_history'], label=model, color=color)
    if plt.gca().get_lines():
        plt.title(f"{title} - Loss History")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"{results_dir}{title.lower()}_loss_history.png",
                    dpi=300, bbox_inches='tight')
    else:
        print(f"No loss history data available for {title}")
    plt.close()

# Function to plot training time comparison


def plot_training_time(data, title):
    models = list(data.keys())
    times = [values.get('training_time', 0) for values in data.values()]
    colors = [assign_color_and_group(model) for model in models]
    architectures = [get_architecture(model) for model in models]

    fig, ax = plt.subplots(figsize=(15, 8))
    bars = ax.bar(models, times, color=colors)

    # Add value labels inside the bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height*0.95,
                f'{height:.2f}',
                ha='center', va='top', rotation=0, color='white', fontweight='bold')

    ax.set_title(f"{title} - Training Time Comparison")
    ax.set_xlabel("Models")
    ax.set_ylabel("Training Time (seconds)")
    plt.xticks(rotation=45, ha='right')

    # Add space between different architectures
    prev_arch = None
    for i, arch in enumerate(architectures):
        if prev_arch and arch != prev_arch:
            ax.axvline(i - 0.5, color='black', linestyle='--', alpha=0.3)
        prev_arch = arch

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Adjust bottom margin
    plt.savefig(f"{results_dir}{title.lower()}_training_time.png",
                dpi=300, bbox_inches='tight')
    plt.close()


# Visualize MNIST results
plot_performance_metrics(mnist_data, "MNIST")
plot_loss_history(mnist_data, "MNIST")
plot_training_time(mnist_data, "MNIST")

# Visualize SUSY results
plot_performance_metrics(susy_data, "SUSY")
plot_loss_history(susy_data, "SUSY")
plot_training_time(susy_data, "SUSY")

print(f"Results have been saved in the '{results_dir}' directory.")
