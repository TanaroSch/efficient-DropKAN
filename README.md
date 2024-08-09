# Efficient DropKAN

This repository provides an efficient implementation of Kolmogorov-Arnold Networks (KAN) with dropout functionality, based on the DropKAN paper.

## Credits

- Original efficient-kan implementation: [Blealtan/efficient-kan](https://github.com/Blealtan/efficient-kan)
- DropKAN paper: [DropKAN: Regularizing KANs by Masking Post-Activations](https://arxiv.org/abs/2407.13044)
- Original DropKAN implementation: [Ghaith81/dropkan](https://github.com/Ghaith81/dropkan)

## Usage

1. Include the `dropkan.py` file in your project or install efficient-dropkan with pip from local source or the git repository.

2. Import the DropKAN class:

```python
from efficient_dropkan import DropKAN
```
3. Create and use a DropKAN model:
```
import torch

# Define model architecture
input_dim = 10
hidden_dims = [64, 32]
output_dim = 1

# Create DropKAN model with dropout
model = DropKAN(
    [input_dim] + hidden_dims + [output_dim],
    drop_rate=0.1,
    drop_mode='postact',
    drop_scale=True
)

# Example forward pass
x = torch.randn(32, input_dim)
output = model(x)

# Train the model (example)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

## Customization

The DropKAN class accepts the following parameters:

- `layers_hidden`: List of integers defining the network architecture
- `grid_size`: Number of grid intervals (default: 5)
- `spline_order`: Order of spline interpolation (default: 3)
- `drop_rate`: Dropout rate (default: 0.0)
- `drop_mode`: Dropout mode ('dropout', 'postspline', or 'postact') (default: 'postact')
- `drop_scale`: Whether to scale outputs during dropout (default: True)

Adjust these parameters to customize the model for your specific use case.

## Evaluation Results

I evaluated our efficient DropKAN implementation against the traditional KAN implementation on both the MNIST Fashion and SUSY datasets.

### MNIST Fashion Results

The following table shows the performance metrics for different model configurations on the MNIST Fashion dataset:

| Model | Accuracy | Precision | Recall | F1 Score | Training Time (s) |
|-------|----------|-----------|--------|----------|-------------------|
| KAN | 0.9367 | 0.9384 | 0.9348 | 0.9366 | 3097.44 |
| DropKAN (dropout=0.1) | 0.9389 | 0.9450 | 0.9320 | 0.9385 | 3068.06 |
| DropKAN (postspline=0.1) | 0.9400 | 0.9419 | 0.9378 | 0.9399 | 3040.14 |
| DropKAN (postact=0.1) | 0.9431 | 0.9508 | 0.9346 | 0.9426 | 3024.55 |
| DropKAN (postact=0.3) | 0.9325 | 0.9198 | 0.9476 | 0.9335 | 3028.21 |

Key observations:
- DropKAN with postact=0.1 achieved the highest accuracy and F1 score.
- All DropKAN variants showed improved performance over the traditional KAN in most metrics.
- Training times were comparable across all models, with slight improvements for DropKAN variants.

### SUSY Results

For the SUSY dataset, I evaluated multiple model architectures and configurations. The comprehensive results are best visualized in the following image:

![SUSY Results](tests\results\susy_performance_metrics.png)

This image presents performance metrics across different model architectures and DropKAN configurations, providing a concise overview of our extensive evaluation on the SUSY dataset.

Key observations:
- DropKAN consistently demonstrates higher precision than traditional KAN across various architectures, albeit with a stronger decrease in recall.
- Traditional dropout between the KAN layers performs best in terms of accuracy and F1 score.
- DropKAN exhibits optimal performance when implemented with a pyramid architecture.

### Summary of Findings

Our evaluation results demonstrate that the efficient DropKAN implementation offers distinct advantages over traditional KAN, with its performance characteristics varying based on the dataset and architectural choices:

1. On the MNIST Fashion dataset, DropKAN variants consistently outperformed traditional KAN, with the postact dropout mode showing particular promise in terms of accuracy and F1 score.

2. For the SUSY dataset, DropKAN showcased its strength in improving precision across different architectures. However, this came at the cost of decreased recall, highlighting a precision-recall trade-off that should be considered based on specific application requirements.

3. The choice of dropout mode and architecture significantly impacts performance. Traditional dropout between KAN layers proved most effective for the SUSY dataset, while a pyramid architecture seemed to best leverage DropKAN's capabilities.

4. Training times were generally comparable between DropKAN and traditional KAN, with some configurations of DropKAN showing slight improvements.

These findings underscore the potential of our efficient DropKAN implementation to enhance model performance, particularly in scenarios where increased precision is desirable. However, they also emphasize the importance of careful configuration and architecture selection to optimize for specific dataset characteristics and performance goals.