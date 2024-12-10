import torch
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from model import UNET

# Load and preprocess the image
def predict_image(image_path, model, device, transform):
    # Load image
    image = np.array(Image.open(image_path).convert("RGB"))

    # Apply transformations
    augmented = transform(image=image)
    image = augmented["image"].unsqueeze(0)  # Add batch dimension

    # Move image to device
    image = image.to(device)

    # Set model to evaluation mode
    model.eval()

    # Perform prediction
    with torch.no_grad():
        pred = torch.sigmoid(model(image))
        pred = (pred > 0.5).float()

    return pred.squeeze(0).cpu().numpy()  # Remove batch dimension and convert to numpy


val_transform = A.Compose(
    [
        A.Resize(height=160, width=240),  # Use the same dimensions as training
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = UNET(in_channels=3, out_channels=1).to(DEVICE)

# Load the trained weights
checkpoint = torch.load("my_checkpoint.pth.tar", map_location=DEVICE)
model.load_state_dict(checkpoint['state_dict'])



# Path to the test image
test_image_path = "data/test_images/example.jpg"

# Predict the mask
predicted_mask = predict_image(test_image_path, model, DEVICE, val_transform)

# Visualize the results
plt.figure(figsize=(10, 5))

# Original image
original_image = np.array(Image.open(test_image_path).convert("RGB"))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(original_image)
plt.axis("off")

# Predicted mask
plt.subplot(1, 2, 2)
plt.title("Predicted Mask")
predicted_mask_to_display = predicted_mask.squeeze(0)  # Shape: (160, 240)
plt.imshow(predicted_mask_to_display, cmap="gray")
plt.axis("off")

plt.show()

