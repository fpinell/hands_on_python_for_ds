import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# Create a dataset of images with and without edges
class EdgeDataset(Dataset):
    def __init__(self, num_samples=1000, image_size=10):
        self.num_samples = num_samples
        self.image_size = image_size
        
        # Pre-allocate arrays for better efficiency
        data = np.zeros((num_samples, image_size, image_size))
        labels = np.zeros(num_samples)
        
        # Generate images with horizontal edges
        for i in range(num_samples // 2):
            # Random position for the edge
            edge_pos = np.random.randint(2, image_size-2)
            # Create image with edge
            data[i, edge_pos:] = 1
            # Add some noise
            data[i] += np.random.normal(0, 0.1, (image_size, image_size))
            labels[i] = 1  # 1 for edge
            
        # Generate images without edges (uniform)
        for i in range(num_samples // 2, num_samples):
            # Create uniform image
            data[i] = np.random.uniform(0, 1, (image_size, image_size))
            labels[i] = 0  # 0 for no edge
            
        # Convert to tensors
        self.data = torch.FloatTensor(data).unsqueeze(1)  # Add channel dimension
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Create dataset and dataloader
dataset = EdgeDataset(num_samples=1000, image_size=10)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Visualize sample images from the dataset
plt.figure(figsize=(15, 5))
plt.suptitle('Sample Images from Dataset', fontsize=16)

# Plot 5 images with edges
for i in range(5):
    plt.subplot(2, 5, i + 1)
    img = dataset.data[i].squeeze().numpy()
    plt.imshow(img, cmap='gray')
    plt.title(f'Edge Image {i+1}')
    plt.axis('off')

# Plot 5 images without edges
for i in range(5):
    plt.subplot(2, 5, i + 6)
    img = dataset.data[500 + i].squeeze().numpy()  # Get images from second half
    plt.imshow(img, cmap='gray')
    plt.title(f'No Edge Image {i+1}')
    plt.axis('off')

plt.tight_layout()
plt.show()

# Define a simple CNN
class EdgeDetectorCNN(nn.Module):
    def __init__(self):
        super(EdgeDetectorCNN, self).__init__()
        # Single convolutional layer with 3x3 kernel
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        # Global average pooling to get a single value per image
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Sigmoid for binary classification
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)  # Reduce to single value per image
        x = self.sigmoid(x)
        return x.squeeze()  # Remove extra dimensions

# Initialize model, loss function, and optimizer
model = EdgeDetectorCNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in dataloader:
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}')

# Visualize the learned filter
learned_filter = model.conv.weight.data.squeeze().numpy()
plt.figure(figsize=(15, 5))

# Plot the learned filter
plt.subplot(131)
plt.imshow(learned_filter, cmap='gray')
plt.title('Learned Filter')
plt.colorbar()

# Plot a sample image with edge
plt.subplot(132)
sample_img = dataset.data[0].squeeze().numpy()  # Get first image (has edge)
plt.imshow(sample_img, cmap='gray')
plt.title('Sample Image with Edge')
plt.colorbar()

# Plot the filter response
plt.subplot(133)
with torch.no_grad():
    response = model.conv(dataset.data[0].unsqueeze(0)).squeeze().numpy()
plt.imshow(response, cmap='gray')
plt.title('Filter Response')
plt.colorbar()

plt.tight_layout()
plt.show()

# Print the filter values
print("\nLearned filter values:")
print(learned_filter) 