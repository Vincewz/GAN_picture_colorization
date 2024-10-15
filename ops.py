import torch
import torch.nn as nn

# Loss functions
def generator_loss(disc_generated_output, gen_output, target):
    """
    Calculate generator loss based on adversarial loss and L1 loss
    """
    adversarial_loss = nn.BCEWithLogitsLoss()(disc_generated_output, torch.ones_like(disc_generated_output))
    l1_loss = nn.L1Loss()(gen_output, target)
    return adversarial_loss + 100 * l1_loss  # Weighted L1 loss

def discriminator_loss(disc_real_output, disc_generated_output):
    """
    Calculate discriminator loss
    """
    real_loss = nn.BCEWithLogitsLoss()(disc_real_output, torch.ones_like(disc_real_output))
    fake_loss = nn.BCEWithLogitsLoss()(disc_generated_output, torch.zeros_like(disc_generated_output))
    return (real_loss + fake_loss) / 2

# Function to convert color images to grayscale
def rgb_to_grayscale(image):
    """
    Convert an image from RGB to grayscale
    """
    return 0.299 * image[:, 0:1, :, :] + 0.587 * image[:, 1:2, :, :] + 0.114 * image[:, 2:3, :, :]

# Function to prepare input for the generator
def prepare_input(image):
    """
    Prepare the input by converting the image to grayscale
    """
    grayscale = rgb_to_grayscale(image)
    return grayscale
