import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# Create a dataset of sequences (same as before)
class SequenceDataset(Dataset):
    def __init__(self, num_samples=1000, seq_length=20):
        self.num_samples = num_samples
        self.seq_length = seq_length
        
        # Pre-allocate numpy arrays for better efficiency
        data = np.zeros((num_samples, seq_length-1))
        targets = np.zeros((num_samples, seq_length-1))
        
        for i in range(num_samples):
            # Random frequency between 0.1 and 0.5
            freq = np.random.uniform(0.1, 0.5)
            # Generate sequence
            t = np.linspace(0, 10, seq_length)
            sequence = np.sin(2 * np.pi * freq * t)
            # Add some noise
            sequence += np.random.normal(0, 0.1, seq_length)
            
            # Input is sequence[:-1], target is sequence[1:]
            data[i] = sequence[:-1]
            targets[i] = sequence[1:]
        
        # Convert to tensors (more efficient now)
        self.data = torch.FloatTensor(data).unsqueeze(-1)  # Add feature dimension
        self.targets = torch.FloatTensor(targets).unsqueeze(-1)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Create dataset and dataloader
dataset = SequenceDataset(num_samples=1000, seq_length=20)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # Batch size 1 for RTRL

# Visualize sample sequences
plt.figure(figsize=(15, 5))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.plot(dataset.data[i].squeeze().numpy(), label='Input')
    plt.plot(dataset.targets[i].squeeze().numpy(), label='Target')
    plt.title(f'Sample Sequence {i+1}')
    plt.legend()
plt.tight_layout()
plt.show()

# Define RNN with RTRL
class RTRLRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, output_size=1, learning_rate=0.01):
        super(RTRLRNN, self).__init__()
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        
        # Initialize weights
        self.Wxh = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)  # Input to hidden
        self.Whh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)  # Hidden to hidden
        self.Why = nn.Parameter(torch.randn(hidden_size, output_size) * 0.01)  # Hidden to output
        
        # Initialize biases
        self.bh = nn.Parameter(torch.zeros(hidden_size))  # Hidden bias
        self.by = nn.Parameter(torch.zeros(output_size))  # Output bias
        
        # Initialize P matrices for RTRL
        self.P = {
            'Wxh': torch.zeros(hidden_size, input_size * hidden_size),
            'Whh': torch.zeros(hidden_size, hidden_size * hidden_size),
            'bh': torch.zeros(hidden_size, hidden_size)
        }
    
    def forward(self, x, hprev=None):
        # x shape: (seq_length, input_size)
        seq_length = x.size(0)
        
        # Initialize hidden state if not provided
        if hprev is None:
            hprev = torch.zeros(self.hidden_size)
        
        # Initialize lists to store values
        hs = []  # Hidden states
        ys = []  # Outputs
        Ps = []  # P matrices
        
        # Process sequence
        for t in range(seq_length):
            # Compute hidden state
            h = torch.tanh(
                torch.matmul(x[t], self.Wxh) +
                torch.matmul(hprev, self.Whh) +
                self.bh
            )
            
            # Compute output
            y = torch.matmul(h, self.Why) + self.by
            
            # Update P matrices for RTRL
            P_new = {}
            
            # Compute derivative of tanh
            dtanh = (1 - h * h).unsqueeze(1)  # Shape: (hidden_size, 1)
            
            # Update P for Wxh
            P_new['Wxh'] = torch.matmul(
                dtanh,
                torch.cat([
                    x[t].unsqueeze(0),  # Shape: (1, input_size)
                    torch.zeros(1, self.hidden_size)  # Padding to match dimensions
                ], dim=1)
            )
            
            # Update P for Whh
            P_new['Whh'] = torch.matmul(
                dtanh,
                torch.cat([
                    hprev.unsqueeze(0),  # Shape: (1, hidden_size)
                    torch.zeros(1, self.hidden_size)  # Padding to match dimensions
                ], dim=1)
            )
            
            # Update P for bh
            P_new['bh'] = torch.matmul(
                dtanh,
                torch.cat([
                    torch.ones(1, 1),  # Shape: (1, 1)
                    torch.zeros(1, self.hidden_size)  # Padding to match dimensions
                ], dim=1)
            )
            
            # Store values
            hs.append(h)
            ys.append(y)
            Ps.append(P_new)
            
            # Update for next time step
            hprev = h
            self.P = P_new
        
        return torch.stack(ys), torch.stack(hs)
    
    def update_weights(self, x, h, y, target, Ps):
        # Compute error
        error = y - target
        
        # Update weights using RTRL
        for t in range(len(x)):
            # Compute gradients
            dWxh = torch.matmul(error[t].unsqueeze(1), Ps[t]['Wxh'])
            dWhh = torch.matmul(error[t].unsqueeze(1), Ps[t]['Whh'])
            dbh = torch.matmul(error[t].unsqueeze(1), Ps[t]['bh'])
            dWhy = torch.matmul(error[t].unsqueeze(1), h[t].unsqueeze(0))
            dby = error[t]
            
            # Update weights
            self.Wxh.data -= self.learning_rate * dWxh
            self.Whh.data -= self.learning_rate * dWhh
            self.bh.data -= self.learning_rate * dbh.squeeze()
            self.Why.data -= self.learning_rate * dWhy
            self.by.data -= self.learning_rate * dby

# Initialize model
model = RTRLRNN(learning_rate=0.01)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, targets in dataloader:
        # Remove batch dimension since we're using batch size 1
        inputs = inputs.squeeze(0)
        targets = targets.squeeze(0)
        
        # Forward pass
        outputs, hidden_states = model(inputs)
        
        # Compute loss
        loss = torch.mean((outputs - targets) ** 2)
        
        # Update weights using RTRL
        model.update_weights(inputs, hidden_states, outputs, targets, model.P)
        
        running_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}')

# Visualize predictions
model.eval()
with torch.no_grad():
    # Get a sample sequence
    sample_input = dataset.data[0]
    sample_target = dataset.targets[0]
    
    # Get model prediction
    prediction, _ = model(sample_input)
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(sample_input.squeeze().numpy(), label='Input')
    plt.plot(sample_target.numpy(), label='Target')
    plt.plot(prediction.squeeze().numpy(), label='Prediction')
    plt.title('RTRL RNN Prediction vs Target')
    plt.legend()
    plt.show()

# Print model parameters
print("\nModel parameters:")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}") 