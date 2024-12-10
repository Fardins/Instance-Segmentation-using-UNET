import streamlit as st
from PIL import Image
import numpy as np
import torch
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import UNET

# Define transformations
val_transform = A.Compose(
    [
        A.Resize(height=160, width=240),  # Ensure same dimensions as training
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ]
)

# Initialize the UNET model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = UNET(in_channels=3, out_channels=1).to(DEVICE)

# Load the model weights
checkpoint = torch.load("my_checkpoint.pth.tar", map_location=DEVICE)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# Prediction function
def predict_image(image, model, device, transform):
    # Apply transformations
    augmented = transform(image=image)
    image = augmented["image"].unsqueeze(0)  # Add batch dimension

    # Move image to device
    image = image.to(device)

    # Perform prediction
    with torch.no_grad():
        pred = torch.sigmoid(model(image))
        pred = (pred > 0.5).float()

    return pred.squeeze(0).cpu().numpy()  # Remove batch dimension and convert to numpy

# Directory containing pre-uploaded images
IMAGE_DIR = "data/test_images"  # Replace with your directory

# Get a list of images from the directory
available_images = [img for img in os.listdir(IMAGE_DIR) if img.endswith(("jpg", "png", "jpeg"))]


# Streamlit app
st.title("UNET Instance Segmentation")

# Allow the user to select an image
selected_image_name = st.selectbox("Select an Image for Prediction", available_images)



if selected_image_name:
    # Path to the selected image
    selected_image_path = os.path.join(IMAGE_DIR, selected_image_name)

    # Display the selected image
    original_image = Image.open(selected_image_path).convert("RGB")
    st.image(original_image, caption=f"Selected Image: {selected_image_name}", use_column_width=True)


    # Convert to numpy array
    image_np = np.array(original_image)

    # Generate prediction
    predicted_mask = predict_image(image_np, model, DEVICE, val_transform)

    # Ensure the mask is grayscale with correct shape
    predicted_mask = (predicted_mask * 255).astype(np.uint8)  # Scale to 0-255
    if len(predicted_mask.shape) == 3 and predicted_mask.shape[0] == 1:
        predicted_mask = predicted_mask.squeeze(0)  # Remove channel dimension

    # Display predicted mask
    st.subheader("Predicted Mask")
    st.image(predicted_mask, caption="Mask Output", use_container_width=True, clamp=True)

