import tensorflow as tf


def generator_loss(disc_generated_output, gen_output, target):
    adversarial_loss = tf.keras.losses.BinaryCrossentropy()(
        tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    return adversarial_loss + 100 * l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = tf.keras.losses.BinaryCrossentropy()(
        tf.ones_like(disc_real_output), disc_real_output)
    fake_loss = tf.keras.losses.BinaryCrossentropy()(
        tf.zeros_like(disc_generated_output), disc_generated_output)
    return (real_loss + fake_loss) / 2

def rgb_to_grayscale(image):
    return tf.image.rgb_to_grayscale(image)