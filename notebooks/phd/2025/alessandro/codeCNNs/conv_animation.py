import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import torch
import torch.nn.functional as F
from matplotlib.widgets import Button

# Create a bigger input image (10x10) with clear horizontal edges
image = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Horizontal edge
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Horizontal edge
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=np.float32)

# Define a horizontal edge detection kernel
kernel = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
], dtype=np.float32)

# Convert to PyTorch tensors
image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
kernel_tensor = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0)

# Create figure and subplots
fig = plt.figure(figsize=(15, 6))  # Reduced height since we removed explanation
gs = fig.add_gridspec(2, 3, height_ratios=[4, 1])  # Removed explanation row
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax_buttons = fig.add_subplot(gs[1, :])

fig.suptitle('Horizontal Edge Detection Animation', fontsize=16)

# Plot original image with pixel values
im1 = ax1.imshow(image, cmap='gray', vmin=0, vmax=1)
ax1.set_title('Input Image')
ax1.set_xticks([])
ax1.set_yticks([])

# Add pixel values with high contrast
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        # Use bright colors for better visibility
        ax1.text(j, i, f'{image[i, j]:.0f}', 
                ha='center', va='center', fontsize=14, weight='bold',
                color='#00FF00' if image[i, j] == 1 else '#FF0000')

# Create a very subtle overlay for kernel coverage
# Adjust rectangle to align with pixel grid
rect = Rectangle((-0.5, -0.5), 3, 3, linewidth=2, edgecolor='#00FFFF', facecolor='#00FFFF', alpha=0.2)
ax1.add_patch(rect)

# Plot kernel with values
im2 = ax2.imshow(kernel, cmap='gray', vmin=-1, vmax=1)
ax2.set_title('Horizontal Edge Kernel')
ax2.set_xticks([])
ax2.set_yticks([])

# Add kernel values with high contrast
for i in range(kernel.shape[0]):
    for j in range(kernel.shape[1]):
        # Use bright colors for better visibility
        color = '#00FF00' if kernel[i, j] > 0 else '#FF0000' if kernel[i, j] < 0 else '#FFFFFF'
        ax2.text(j, i, f'{kernel[i, j]:.0f}', 
                ha='center', va='center', fontsize=14, weight='bold',
                color=color)

# Plot output
im3 = ax3.imshow(np.zeros_like(image), cmap='gray', vmin=-3, vmax=3)
ax3.set_title('Output Feature Map')
ax3.set_xticks([])
ax3.set_yticks([])

# Initialize output feature map
output = np.zeros_like(image)

# Add buttons
play_pause_ax = plt.axes([0.3, 0.05, 0.15, 0.075])
play_pause_button = Button(play_pause_ax, 'Play/Pause', color='lightgray', hovercolor='0.975')

step_ax = plt.axes([0.5, 0.05, 0.15, 0.075])
step_button = Button(step_ax, 'Step', color='lightgray', hovercolor='0.975')

reset_ax = plt.axes([0.7, 0.05, 0.15, 0.075])
reset_button = Button(reset_ax, 'Reset', color='lightgray', hovercolor='0.975')

# Animation control
is_playing = True
current_frame = 0

def step(event):
    global current_frame
    update(None)
    plt.draw()

def play_pause(event):
    global is_playing
    is_playing = not is_playing
    if is_playing:
        anim.event_source.start()
    else:
        anim.event_source.stop()

def reset(event):
    global output, current_frame, is_playing
    output = np.zeros_like(image)
    current_frame = 0
    is_playing = False
    im1.set_array(image)
    im3.set_array(output)
    rect.set_xy((-0.5, -0.5))  # Reset rectangle position
    plt.draw()

play_pause_button.on_clicked(play_pause)
step_button.on_clicked(step)
reset_button.on_clicked(reset)

def update(frame):
    global current_frame, is_playing
    if not is_playing and frame is not None:
        return im1, im3
    # Calculate current position
    row = current_frame // (image.shape[1] - kernel.shape[1] + 1)
    col = current_frame % (image.shape[1] - kernel.shape[1] + 1)
    # Move rectangle to highlight kernel coverage (adjusted for pixel grid)
    rect.set_xy((col - 0.5, row - 0.5))
    # Calculate the convolution at this position
    conv_value = 0
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            pixel_value = image[row+i, col+j]
            kernel_value = kernel[i, j]
            conv_value += pixel_value * kernel_value
    # Update only the current position in the output
    output[row, col] = conv_value
    # Update the output visualization
    ax3.clear()
    ax3.imshow(output, cmap='gray', vmin=-3, vmax=3)
    ax3.set_title('Output Feature Map')
    ax3.set_xticks([])
    ax3.set_yticks([])
    # Show only the values that have been computed so far
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if (i < row) or (i == row and j <= col):
                value = output[i, j]
                if value > 1:
                    color = '#00FF00'
                elif value < -1:
                    color = '#FF0000'
                elif value != 0:
                    color = '#FFFFFF'
                else:
                    color = '#FFFF00'
                ax3.text(j, i, f'{value:.1f}', 
                        ha='center', va='center', fontsize=14, weight='bold',
                        color=color)
    current_frame += 1
    if current_frame >= (image.shape[0] - kernel.shape[0] + 1) * (image.shape[1] - kernel.shape[1] + 1):
        current_frame = 0
    return im1, im3

# Create animation
anim = FuncAnimation(
    fig, 
    update, 
    frames=None,  # We'll control frames manually
    interval=1000,  # Slower animation to read the values
    blit=False  # Changed to False to allow text updates
)

plt.tight_layout()
plt.show() 