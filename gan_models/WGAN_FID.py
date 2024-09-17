!pip install google-colab

!pip install PyDrive

pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib 

from pydrive.auth import GoogleAuth 

from pydrive.drive import GoogleDrive 
# Initialize GoogleAuth 
gauth = GoogleAuth() 
gauth.LoadClientConfigFile("client_secret_465500413129-0vtor8uqh740qvuh1f8idjg24923eovo.apps.googleusercontent.com.json") 

# Authenticate, if required 

if gauth.credentials is None: 

    # Authenticate if they're not there 

    gauth.LocalWebserverAuth() 

elif gauth.access_token_expired: 

    # Refresh them if expired 

    gauth.Refresh() 

else: 

    # Initialize the saved creds 

    gauth.Authorize() 

  

# Create GoogleDrive instance with authenticated GoogleAuth instance 

drive = GoogleDrive(gauth) 

  

# List files in the root of your Google Drive 

file_list = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList() 

for file1 in file_list: 

    print('title: %s, id: %s' % (file1['title'], file1['id'])) 

!pip install PyDrive

from google.colab import drive
drive.mount('/gdrive')

!unzip img_align_celeba.zip

!ls /gdrive/MyDrive/data/CelebA/Img

!pip install comet_ml

!pip install torch

from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

experiment = Experiment(
  api_key="t0zqSwN0yEbycxLxznUqLTkHU",
  project_name="wgan-with-fr-chet-inception-distance",
  workspace="gdi1u21"
)

!pip install torchvision

!pip install gdown
!pip install requests
!pip install PIL
!pip install natsort
!pip install numpy
!pip install scipy.linalg
!pip install matplotlib.pyplot

import zipfile 

import os 

  

zip_path = 'img_align_celeba.zip'  # Path to the downloaded zip file 

extract_to = './CelebA'  # Directory where files will be extracted 

  

# Create target directory if it doesn't exist 

if not os.path.exists(extract_to): 

    os.makedirs(extract_to) 

  

# Unzipping the file 

with zipfile.ZipFile(zip_path, 'r') as zip_ref: 

    zip_ref.extractall(extract_to) 

  

print(f"Files extracted to {extract_to}") 

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
from torchvision.models import inception_v3
import numpy as np
from scipy.linalg import sqrtm
import torch.nn.functional as F
import matplotlib.pyplot as plt
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

def get_inception_features(model, images, device):
    model.eval()
    features = []
    with torch.no_grad():
        for batch in DataLoader(images, batch_size=32):
            batch = batch.to(device)
            # Get output from the global average pooling layer or pre-softmax layer
            feat = model(batch)
            if isinstance(feat, tuple):
                feat = feat[-1]  # Extract features from the tuple
            feat = feat.squeeze()  # Remove single-dimensional entries
            if feat.dim() == 1:
                feat = feat.unsqueeze(0)  # Ensure 2D shape for single image batch
            features.append(feat.cpu().numpy())
    features = np.concatenate(features, axis=0)
    return features

def get_features(model, dataloader, device):
    model.eval()
    features = []
    with torch.no_grad():
        for images in dataloader:
            if isinstance(images, tuple) or isinstance(images, list):
                images = images[0]
            images = images.to(device)
            feat = model(images)
            features.append(feat.cpu().numpy())
    features = np.concatenate(features, axis=0)
    return features

def calculate_fid(real_features, fake_features):
    # Ensure that the feature arrays are 2D
    if real_features.ndim != 2 or fake_features.ndim != 2:
        raise ValueError("Feature arrays must be 2D")

    # Calculate the mean and covariance of the features
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)

    # Calculate the sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)

    # Compute the square root of the product of covariances
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Compute the FID score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

## Create a custom Dataset class
class CelebADataset(Dataset):
  def __init__(self, root_dir, transform=None):
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
img_folder = './celeba/img_align_celeba'  # Update path based on where files got extracted 
# Path to folder with the dataset
dataset_folder = 'img_align_celeba'
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

# Model, Device, and Hyperparameters Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 5e-5
BATCH_SIZE = 64
Z_DIM = 100
NUM_EPOCHS = 10
FEATURES_DISC = 64
FEATURES_GEN = 64
CHANNELS_IMG = 3
CRITIC_ITERATIONS = 5
WEIGHT_CLIP = 0.01

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

# Load the dataset from file and apply transformations
celeba_dataset = CelebADataset(img_folder, transform)

# Data Preparation
transforms = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(3)], [0.5 for _ in range(3)])
])
celeba_dataset = CelebADataset(img_folder, transform) 
dataset = CelebADataset('img_align_celeba', transform=transforms)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize Inception Model for FID
inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
inception_model.eval()

# Training Loop
fixed_noise = torch.randn(32, Z_DIM, 1, 1, device=device)
fid_scores = []
n_epochs_for_fid = 1

for epoch in range(NUM_EPOCHS):
    for batch_idx, real in enumerate(celeba_dataloader):
        real = real.to(device)

        # Training the Critic
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(device)
            fake = gen(noise)

            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))

            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

            # Clipping the weights of the discriminator
            for p in critic.parameters():
                p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

        # Training the Generator
        output = critic(fake).reshape(-1)
        loss_gen = -torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Logging the losses and visualizing the results
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(celeba_dataloader)} "
                  f"Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}")

            with torch.no_grad():
                fake = gen(fixed_noise)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                # Optional: Save or display images

    if epoch % n_epochs_for_fid == 0:
        with torch.no_grad():
            # Generate fake images and ensure they have the correct dimensions
            fake_images = gen(fixed_noise).to(device)
            if len(fake_images.shape) == 4:  # Shape should be (N, C, H, W)
                fake_images_resized = F.interpolate(fake_images, size=(299, 299), mode='bilinear', align_corners=False)
                fake_features = get_inception_features(inception_model, fake_images_resized, device)
            else:
                continue  # Skip if dimensions are incorrect
            print(f"Fake features shape: {fake_features.shape}")
            print(f"Shape of fake images after resizing: {fake_images_resized.shape}")

            # Get real images and ensure they have the correct dimensions
            real_batch = next(iter(celeba_dataloader))
            real_images = real_batch[0].to(device) if isinstance(real_batch, (tuple, list)) else real_batch.to(device)
            if len(real_images.shape) == 4:  # Shape should be (N, C, H, W)
                real_images_resized = F.interpolate(real_images, size=(299, 299), mode='bilinear', align_corners=False)
                real_features = get_inception_features(inception_model, real_images_resized, device)
            else:
                continue  # Skip if dimensions are incorrect
            print(f"Real features shape: {real_features.shape}")
            print(f"Shape of real images after resizing: {real_images_resized.shape}")

            # Calculate FID
            try:
                fid_value = calculate_fid(real_features, fake_features)
                fid_scores.append(fid_value)
                print(f"FID at epoch {epoch}: {fid_value}")
            except Exception as e:
                print(f"Error calculating FID: {e}")
                continue  # Skip to next epoch or batch



# Plotting FID Scores after training
plt.figure(figsize=(10, 5))
if len(fid_scores) > 0:
    plt.plot(range(0, NUM_EPOCHS, n_epochs_for_fid), fid_scores, marker='o')
    plt.title("FID Score over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("FID Score")
    plt.grid(True)
    plt.show()
else:
    print("No FID scores to plot.")

