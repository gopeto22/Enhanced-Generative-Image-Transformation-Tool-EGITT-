from google.colab import drive
drive.mount('/gdrive')

!unzip -q /gdrive/MyDrive/data/CelebA/Img/img_align_celeba.zip

!ls /gdrive/MyDrive/data/CelebA/Img

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import gdown
import requests
from PIL import Image
from natsort import natsorted
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

#from torch.utils.tensorboard import SummaryWriter
#from model import Discriminator, Generator, initialize_weights

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(
                channels_img, features_d, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
        )

        # Move the discriminator to the same device as the model's parameters
        self.disc = self.disc.to(device)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import gdown
import requests
from PIL import Image
from natsort import natsorted
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

#from torch.utils.tensorboard import SummaryWriter
#from model import Discriminator, Generator, initialize_weights

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(
                channels_img, features_d, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
        )

        # Move the discriminator to the same device as the model's parameters
        self.disc = self.disc.to(device)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator,self).__init__()
        self.gen = nn.Sequential(
            self._block(z_dim, features_g * 16, 4, 1, 0),
            self._block(features_g * 16, features_g * 8, 4, 2, 1),
            self._block(features_g * 8, features_g * 4, 4, 2, 1),
            self._block(features_g * 4, features_g * 2, 4, 2, 1),
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size = 4, stride = 2,padding = 1
            ),
            nn.Tanh()
        )

        # Move the generator to the same device as the model's parameters
        self.gen = self.gen.to(device)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias = False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.gen(x)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
def test():
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    initialize_weights(disc)
    assert disc(x).shape == (N, 1, 1, 1 )
    initialize_weights(gen)
    gen = Generator(z_dim, in_channels, 8)
    z = torch.randn((N, z_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W)

LEARNING_RATE = 5e-5
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 1
NUM_EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64
CRITIC_ITERATIONS = 5
WEIGHT_CLIP = 0.01

# Number of gpus available
ngpu = 1
device = torch.device('cuda:0' if (
    torch.cuda.is_available() and ngpu > 0) else 'cpu')

# Path to folder with the dataset
dataset_folder = 'img_align_celeba'

## Create a custom Dataset class
class CelebADataset(Dataset):
  def __init__(self, root_dir, transform=None):
    """
    Args:
      root_dir (string): Directory with all the images
      transform (callable, optional): transform to be applied to each image sample
    """
    # Read names of images in the root directory
    image_names = os.listdir(root_dir)

    self.root_dir = root_dir
    self.transform = transform
    self.image_names = natsorted(image_names)

  def __len__(self):
    return len(self.image_names)

  def __getitem__(self, idx):
    # Get the path to the image
    img_path = os.path.join(self.root_dir, self.image_names[idx])
    # Load image and convert it to RGB
    img = Image.open(img_path).convert('RGB')
    # Apply transformations to the image
    if self.transform:
      img = self.transform(img)

    return img

## Load the dataset
# Path to directory with all the images
img_folder = f'{dataset_folder}'
# Spatial size of training images, images are resized to this size.
image_size = 64
# Transformations to be applied to each individual image sample
transform=transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                          std=[0.5, 0.5, 0.5])
])
# Load the dataset from file and apply transformations
celeba_dataset = CelebADataset(img_folder, transform)

## Create a dataloader
# Batch size during training
batch_size = 128
# Number of workers for the dataloader
num_workers = 0 if device.type == 'cuda' else 2
# Whether to put fetched data tensors to pinned memory
pin_memory = True if device.type == 'cuda' else False

celeba_dataloader = torch.utils.data.DataLoader(celeba_dataset,
                                                batch_size=batch_size,
                                                num_workers=num_workers,
                                                pin_memory=pin_memory,
                                                shuffle=True)

#dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN)
critic = Discriminator(CHANNELS_IMG, FEATURES_DISC)
initialize_weights(gen)
initialize_weights(critic)

opt_gen = optim.RMSprop(gen.parameters(), lr = LEARNING_RATE)
opt_critic = optim.RMSprop(critic.parameters(), lr = LEARNING_RATE)

fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
#writer_real = SummaryWriter(f"logs/real")
#writer_real = SummaryWriter(f"logs/fake")
gen.train()
critic.train()

step = 0



for epoch in range(NUM_EPOCHS):
  for batch_idx, real in enumerate(celeba_dataloader):
    real = real.to(device)


    for _ in range(CRITIC_ITERATIONS):
      noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(device)
      fake = gen(noise)
      critic_real = critic(real).reshape(-1)
      critic_fake = critic(fake).reshape(-1)
      loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))
      critic.zero_grad()
      loss_critic.backward(retain_graph = True)
      opt_critic.step()

      for p in critic.parameters():
        p.data.clamp(-WEIGHT_CLIP, WEIGHT_CLIP)

    output = critic(fake).reshape(-1)
    loss_gen = -torch.mean(output)
    opt_gen.zero_grad()
    loss_gen.backward()
    opt_gen.step()

    if batch_idx % 100 == 0:
      print(
          f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(celeba_dataloader)} \
          Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
      )


      with torch.no_grad():
        fake = gen(fixed_noise)
        img_grid_real = torchvision.utils.make_grid(
            real[:32], normalize = True
        )
        img_grid_fake = torchvision.utils.make_grid(
            fake[:32], normalize = True
        )
        #writer_real.add_image("Real ", img_grid_real, global_step = step)
        #writer_real.add_image("Fake ", img_grid_fake, global_step = step)
        step += 1

        )
    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator,self).__init__()
        self.gen = nn.Sequential(
            self._block(z_dim, features_g * 16, 4, 1, 0),
            self._block(features_g * 16, features_g * 8, 4, 2, 1),
            self._block(features_g * 8, features_g * 4, 4, 2, 1),
            self._block(features_g * 4, features_g * 2, 4, 2, 1),
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size = 4, stride = 2,padding = 1
            ),
            nn.Tanh()
        )

        # Move the generator to the same device as the model's parameters
        self.gen = self.gen.to(device)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias = False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.gen(x)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
def test():
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    initialize_weights(disc)
    assert disc(x).shape == (N, 1, 1, 1 )
    initialize_weights(gen)
    gen = Generator(z_dim, in_channels, 8)
    z = torch.randn((N, z_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W)

LEARNING_RATE = 5e-5
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 1
NUM_EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64
CRITIC_ITERATIONS = 5
WEIGHT_CLIP = 0.01

# Number of gpus available
ngpu = 1
device = torch.device('cuda:0' if (
    torch.cuda.is_available() and ngpu > 0) else 'cpu')

# Path to folder with the dataset
dataset_folder = 'img_align_celeba'

## Create a custom Dataset class
class CelebADataset(Dataset):
  def __init__(self, root_dir, transform=None):
    """
    Args:
      root_dir (string): Directory with all the images
      transform (callable, optional): transform to be applied to each image sample
    """
    # Read names of images in the root directory
    image_names = os.listdir(root_dir)

    self.root_dir = root_dir
    self.transform = transform
    self.image_names = natsorted(image_names)

  def __len__(self):
    return len(self.image_names)

  def __getitem__(self, idx):
    # Get the path to the image
    img_path = os.path.join(self.root_dir, self.image_names[idx])
    # Load image and convert it to RGB
    img = Image.open(img_path).convert('RGB')
    # Apply transformations to the image
    if self.transform:
      img = self.transform(img)

    return img

## Load the dataset
# Path to directory with all the images
img_folder = f'{dataset_folder}'
# Spatial size of training images, images are resized to this size.
image_size = 64
# Transformations to be applied to each individual image sample
transform=transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                          std=[0.5, 0.5, 0.5])
])
# Load the dataset from file and apply transformations
celeba_dataset = CelebADataset(img_folder, transform)

## Create a dataloader
# Batch size during training
batch_size = 128
# Number of workers for the dataloader
num_workers = 0 if device.type == 'cuda' else 2
# Whether to put fetched data tensors to pinned memory
pin_memory = True if device.type == 'cuda' else False

celeba_dataloader = torch.utils.data.DataLoader(celeba_dataset,
                                                batch_size=batch_size,
                                                num_workers=num_workers,
                                                pin_memory=pin_memory,
                                                shuffle=True)

#dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN)
critic = Discriminator(CHANNELS_IMG, FEATURES_DISC)
initialize_weights(gen)
initialize_weights(critic)

opt_gen = optim.RMSprop(gen.parameters(), lr = LEARNING_RATE)
opt_critic = optim.RMSprop(critic.parameters(), lr = LEARNING_RATE)

fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
#writer_real = SummaryWriter(f"logs/real")
#writer_real = SummaryWriter(f"logs/fake")
gen.train()
critic.train()

step = 0



for epoch in range(NUM_EPOCHS):
  for batch_idx, real in enumerate(celeba_dataloader):
    real = real.to(device)


    for _ in range(CRITIC_ITERATIONS):
      noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(device)
      fake = gen(noise)
      critic_real = critic(real).reshape(-1)
      critic_fake = critic(fake).reshape(-1)
      loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))
      critic.zero_grad()
      loss_critic.backward(retain_graph = True)
      opt_critic.step()

      for p in critic.parameters():
        p.data.clamp(-WEIGHT_CLIP, WEIGHT_CLIP)

    output = critic(fake).reshape(-1)
    loss_gen = -torch.mean(output)
    opt_gen.zero_grad()
    loss_gen.backward()
    opt_gen.step()

    if batch_idx % 100 == 0:
      print(
          f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(celeba_dataloader)} \
          Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
      )


      with torch.no_grad():
        fake = gen(fixed_noise)
        img_grid_real = torchvision.utils.make_grid(
            real[:32], normalize = True
        )
        img_grid_fake = torchvision.utils.make_grid(
            fake[:32], normalize = True
        )
        #writer_real.add_image("Real ", img_grid_real, global_step = step)
        #writer_real.add_image("Fake ", img_grid_fake, global_step = step)
        step += 1


import matplotlib.pyplot as plt
import torchvision.utils as vutils

# Number of images to generate for visualization
num_vis_images = 32

# Generate some fake images with the trained generator
with torch.no_grad():
    gen.eval()  # Set the generator to evaluation mode
    generated_images = gen(fixed_noise).detach().cpu()

# Create a grid of images and plot
img_grid_fake = vutils.make_grid(generated_images[:num_vis_images], normalize=True)
plt.imshow(img_grid_fake.permute(1, 2, 0))
plt.axis("off")
plt.show()


import matplotlib.pyplot as plt


for epoch in range(NUM_EPOCHS):
    for batch_idx, real in enumerate(celeba_dataloader):
        real = real.to(device)

        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(device)
            fake = gen(noise)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

            for p in critic.parameters():
                p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

        output = critic(fake).reshape(-1)
        loss_gen = -torch.mean(output)
        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(celeba_dataloader)} \
                Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                img_grid_real = torchvision.utils.make_grid(
                    real[:32], normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:32], normalize=True
                )

                # Display the images using matplotlib
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.imshow(img_grid_real.permute(1, 2, 0).cpu().numpy())
                plt.title('Real Images')
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.imshow(img_grid_fake.permute(1, 2, 0).cpu().numpy())
                plt.title('Generated Images')
                plt.axis('off')

                plt.show()

            step += 1


