import torch
from torchvision import transforms
from PIL import Image
import os
from network import UNetGenerator
from ops import prepare_input

# Directory to save the results
RESULTS_DIR = 'results/'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load the trained generator model
MODEL_PATH = 'saved_models/generator_epoch_5.pth'  # Example: load the model from epoch 5
generator = UNetGenerator()
generator.load_state_dict(torch.load(MODEL_PATH))
generator.eval()  # Set the model to evaluation mode

# Load a test image
image_path = 'dataset/00001.jpg'  # Path to the image to test
image = Image.open(image_path)

# Transform the image
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor()
])
image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Prepare input by converting to grayscale
grayscale_image = prepare_input(image_tensor)

# Pass the grayscale image through the generator to get the colorized image
with torch.no_grad():
    colorized_image = generator(grayscale_image)

# Convert the generated tensor back to an image format
colorized_image = colorized_image.squeeze(0).permute(1, 2, 0).cpu()
colorized_image = (colorized_image * 255).numpy().astype('uint8')  # Scale back to [0, 255]

# Save the colorized image to the results directory
output_image_path = os.path.join(RESULTS_DIR, f"colorized_{os.path.basename(image_path)}")
Image.fromarray(colorized_image).save(output_image_path)

print(f"Colorized image saved at: {output_image_path}")