import tensorflow as tf
from PIL import Image
import numpy as np
import os
from network import UNetGenerator

# Configurations
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

print("\nAvailable files in saved_models:")
for file in os.listdir('saved_models'):
    print(f"- {file}")

generator = UNetGenerator()

dummy_input = tf.random.normal([1, 256, 256, 1])
_ = generator(dummy_input)

generator.load_weights('saved_models/generator_epoch_2.h5')
print("Model weights loaded successfully")

def load_and_preprocess_image(image_path):
    # Load image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    
    # Resize
    image = tf.image.resize(image, [256, 256])
    # Normalize
    image = tf.cast(image, tf.float32) / 255.0
    return image

try:
    # Load and process test image
    image_path = 'test3.jpg'
    image = load_and_preprocess_image(image_path)
    image = tf.expand_dims(image, 0)  # Add batch dimension

    # Convert to grayscale
    grayscale_image = tf.image.rgb_to_grayscale(image)

    # Save grayscale image
    grayscale_image_np = tf.squeeze(grayscale_image).numpy() * 255.0
    grayscale_image_pil = Image.fromarray(np.uint8(grayscale_image_np))
    grayscale_path = os.path.join(RESULTS_DIR, f"grayscale_{os.path.basename(image_path)}")
    grayscale_image_pil.save(grayscale_path)
    print(f"Grayscale image saved at: {grayscale_path}")

    # Generate colorized image
    colorized_image = generator(grayscale_image, training=False)

    # Convert to PIL image and save
    colorized_image_np = tf.squeeze(colorized_image).numpy() * 255.0
    colorized_image_pil = Image.fromarray(np.uint8(colorized_image_np))
    output_path = os.path.join(RESULTS_DIR, f"colorized_{os.path.basename(image_path)}")
    colorized_image_pil.save(output_path)
    print(f"Colorized image saved at: {output_path}")

except Exception as e:
    print(f"\nError during execution: {e}")