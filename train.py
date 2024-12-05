import tensorflow as tf
import os
import datetime
import csv
from network import UNetGenerator, CNNDiscriminator
from ops import generator_loss, discriminator_loss, rgb_to_grayscale

# Configurations
# BATCH_SIZE = 16
# EPOCHS = 50
# LEARNING_RATE_GEN = 2e-4
# LEARNING_RATE_DISC = 1e-4
# BETA1 = 0.5
# BETA2 = 0.999

# BATCH_SIZE = 16
# EPOCHS = 4
# LEARNING_RATE_GEN = 2e-4
# LEARNING_RATE_DISC = 2e-4
# BETA1 = 0.5
# BETA2 = 0.5

BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE_GEN = 2e-4
LEARNING_RATE_DISC = 5e-5
BETA1 = 0.5
BETA2 = 0.5

class ImageDataset:
    def __init__(self, root_dir, start=1, end=100, batch_size=16):
        self.image_paths = [
            os.path.join(root_dir, f"{i:05d}.jpg") 
            for i in range(start, end+1)
        ]
        self.dataset = tf.data.Dataset.from_tensor_slices(self.image_paths)
        self.dataset = self.dataset.map(self.load_and_preprocess_image)
        self.dataset = self.dataset.batch(batch_size)
        self.dataset = self.dataset.prefetch(tf.data.AUTOTUNE)

    def load_and_preprocess_image(self, path):
        # Lire l'image
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image)
        # Redimensionner
        image = tf.image.resize(image, [256, 256])
        # Normaliser
        image = tf.cast(image, tf.float32) / 255.0
        return image

@tf.function
def train_step(generator, discriminator, gen_optimizer, disc_optimizer,
               real_images, grayscale_images):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate images
        generated_images = generator(grayscale_images, training=True)
        
        # Prepare discriminator data
        real_input = tf.concat([grayscale_images, real_images], axis=3)
        fake_input = tf.concat([grayscale_images, generated_images], axis=3)
        
        # discriminator prediction
        real_output = discriminator(real_input, training=True)
        fake_output = discriminator(fake_input, training=True)
        
        # Calculate loss
        gen_loss = generator_loss(fake_output, generated_images, real_images)
        disc_loss = discriminator_loss(real_output, fake_output)
    
    # Calculate gradient
    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))
    
    return gen_loss, disc_loss

def main():
   # Create necessary directories
   os.makedirs('saved_models', exist_ok=True)
   os.makedirs('logs', exist_ok=True)

   # Create output CSV file
   csv_path = 'training_logs.csv'
   
   with open(csv_path, 'w', newline='') as csvfile:
       csv_writer = csv.writer(csvfile)
       csv_writer.writerow(['Epoch', 'Step', 'Generator_Loss', 'Discriminator_Loss'])

   dataset = ImageDataset('dataset', start=1, end=15000, batch_size=BATCH_SIZE)

   # Initialize models
   generator = UNetGenerator()
   discriminator = CNNDiscriminator()

   # Optimizers 
   gen_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE_GEN, beta_1=BETA1, beta_2=BETA2)
   disc_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE_DISC, beta_1=BETA1, beta_2=BETA2)

   try:
       for epoch in range(EPOCHS):
           for i, batch in enumerate(dataset.dataset):
               # Convert to grayscale
               grayscale_batch = rgb_to_grayscale(batch)
               
               # Training step
               gen_loss, disc_loss = train_step(
                   generator, discriminator,
                   gen_optimizer, disc_optimizer,
                   batch, grayscale_batch
               )

               # Save losses every 100 steps
               if i % 100 == 0:
                   with open(csv_path, 'a', newline='') as csvfile:
                       csv_writer = csv.writer(csvfile)
                       csv_writer.writerow([
                           epoch + 1,
                           i,
                           float(gen_loss),
                           float(disc_loss)
                       ])
                   
                   print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i}], "
                         f"Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}")
           
           # Save models after each epoch
           generator.save(f'saved_models/generator_epoch_{epoch+1}.h5')
           discriminator.save(f'saved_models/discriminator_epoch_{epoch+1}.h5')
           print(f"Models saved at epoch {epoch+1}")

   except Exception as e:
       print(f"An error occurred: {e}")
       generator.save('saved_models/generator_backup.h5') 
       discriminator.save('saved_models/discriminator_backup.h5')
       raise e

if __name__ == '__main__':
   main()