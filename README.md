# Efficient DropKAN

This repository provides an efficient implementation of Kolmogorov-Arnold Networks (KAN) with dropout functionality, based on the DropKAN paper.

## Credits

- Original efficient-kan implementation: [Blealtan/efficient-kan](https://github.com/Blealtan/efficient-kan)
- DropKAN paper: [DropKAN: Regularizing KANs by Masking Post-Activations](https://arxiv.org/abs/2407.13044)
- Original DropKAN implementation: [Ghaith81/dropkan](https://github.com/Ghaith81/dropkan)

## Usage

1. Include the `kan.py` file in your project or install efficient-dropkan with pip from local source or the git repository.

2. Import the KAN class:

```python
from efficient_kan import KAN
```
3. Create and use a KAN model:
```
import torch

# Define model architecture
input_dim = 10
hidden_dims = [64, 32]
output_dim = 1

# Create KAN model with dropout
model = KAN(
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

The KAN class accepts the following parameters:

- `layers_hidden`: List of integers defining the network architecture
- `grid_size`: Number of grid intervals (default: 5)
- `spline_order`: Order of spline interpolation (default: 3)
- `drop_rate`: Dropout rate (default: 0.0)
- `drop_mode`: Dropout mode ('dropout', 'postspline', or 'postact') (default: 'postact')
- `drop_scale`: Whether to scale outputs during dropout (default: True)

Adjust these parameters to customize the model for your specific use case.
