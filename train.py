import multiprocessing
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
from torchvision import transforms
from network import UNetGenerator, CNNDiscriminator
from ops import generator_loss, discriminator_loss, prepare_input
import os
from torch.utils.tensorboard import SummaryWriter

class ImageDataset(Dataset):
    def __init__(self, root_dir, start=1, end=100, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f"{i:05d}.jpg" for i in range(start, end+1)]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image

def main():
    # Utiliser CPU ou GPU si disponible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    writer = SummaryWriter('runs/GAN_training')

    # Paramètres d'entraînement
    BATCH_SIZE = 16
    EPOCHS = 50
    LR = 2e-4
    NUM_WORKERS = 0  # Set to 0 for debugging, increase if no issues occur

    # Chargement du dataset
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = ImageDataset(root_dir='dataset', start=1, end=15000, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    # Initialiser les réseaux
    generator = UNetGenerator().to(device)
    discriminator = CNNDiscriminator().to(device)

    # Visualisation des modèles avec une vraie image du dataset
    real_image = next(iter(dataloader)).to(device)
    grayscale_image = prepare_input(real_image).to(device)
    writer.add_graph(generator, grayscale_image)
    writer.add_graph(discriminator, torch.cat((grayscale_image, real_image), 1))

    # Optimizers
    gen_optimizer = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

    # Boucle d'entraînement
    for epoch in range(EPOCHS):
        for i, real_images in enumerate(dataloader):
            real_images = real_images.to(device)

            # Convertir les images en niveaux de gris
            grayscale_images = prepare_input(real_images).to(device)

            # Entraîner le Discriminateur
            gen_images = generator(grayscale_images)
            
            # S'assurer que toutes les tailles sont les mêmes
            if gen_images.size() != real_images.size():
                gen_images = torch.nn.functional.interpolate(gen_images, size=real_images.size()[2:])
            
            real_input = torch.cat((grayscale_images, real_images), 1)
            fake_input = torch.cat((grayscale_images, gen_images.detach()), 1)

            real_output = discriminator(real_input)
            fake_output = discriminator(fake_input)

            disc_loss = discriminator_loss(real_output, fake_output)
            disc_optimizer.zero_grad()
            disc_loss.backward()
            disc_optimizer.step()

            # Entraîner le Générateur
            fake_input = torch.cat((grayscale_images, gen_images), 1)
            fake_output = discriminator(fake_input)
            gen_loss = generator_loss(fake_output, gen_images, real_images)
            gen_optimizer.zero_grad()
            gen_loss.backward()
            gen_optimizer.step()

            # Afficher les pertes et les ajouter à TensorBoard
            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i}/{len(dataloader)}], "
                      f"Gen Loss: {gen_loss.item():.4f}, Disc Loss: {disc_loss.item():.4f}")
                writer.add_scalar('Loss/Generator', gen_loss.item(), epoch * len(dataloader) + i)
                writer.add_scalar('Loss/Discriminator', disc_loss.item(), epoch * len(dataloader) + i)

        # Sauvegarder les modèles à chaque epoch
        torch.save(generator.state_dict(), os.path.join('saved_models', f'generator_epoch_{epoch+1}.pth'))
        torch.save(discriminator.state_dict(), os.path.join('saved_models', f'discriminator_epoch_{epoch+1}.pth'))
        print(f"Models saved at epoch {epoch+1}")

    writer.close()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
