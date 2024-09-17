import torch


class TransformerNet(torch.nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out

!pip install av

from torchvision import transforms
import torch
import numpy as np
import av

# Mean and standard deviation used for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def extract_frames(video_path):
    """ Extracts frames from video """
    frames = []
    video = av.open(video_path)
    for frame in video.decode(0):
        yield frame.to_image()


def gram_matrix(y):
    """ Returns the gram matrix of y (used to compute style loss) """
    (b, c, h, w) = y.size()
    features = y.view(b, c, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (c * h * w)
    return gram


def train_transform(image_size):
    """ Transforms for training images """
    transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.RandomCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return transform


def style_transform(image_size=None):
    """ Transforms for style image """
    resize = [transforms.Resize(image_size)] if image_size else []
    transform = transforms.Compose(resize + [transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transform


def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return tensors

import time
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import torch
# Get the VGG model
import torchvision.models.vgg as vgg
import torch.utils.model_zoo as model_zoo
# Image tranformation pipeline
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.optim import Adam
from PIL import Image,ImageFile
from tqdm import tqdm,tqdm_notebook
#from fast_neural_style.transformer_net import TransformerNet
#from fast_neural_style.utils import gram_matrix, recover_image, tensor_normalizer
%matplotlib inline
ImageFile.LOAD_TRUNCATED_IMAGES = True

!pip install torch

SEED = 999
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available:
  torch.cuda.manual_seed(SEED)
  kwargs = {'num_workers': 4, 'pin_memory': True}
else:
  kwargs = {}

class TensorNormalizer(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def __call__(self, tensor):
        """
        Normalize a tensor image with mean and standard deviation.
        """
        return (tensor - self.mean) / self.std

IMAGE_SIZE = 256
BATCH_SIZE = 4
DATASET = '/content/drive/MyDrive/data_coco'
# Define mean and std
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])

# Instantiate your custom normalizer
tensor_normalizer = TensorNormalizer(mean, std)

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    tensor_normalizer  # Use the instance of your custom class
])

train_dataset = datasets.ImageFolder(DATASET, transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)


LossOutput = namedtuple("LossOutput", ["relu1_2","relu2_2", "relu3_3", "relu4_3"])

class LossNetwork(torch.nn.Module):
  def __init__(self,vgg_model):
    super(LossNetwork, self).__init__()
    self.vgg_layers = vgg_model.features
    self.layer_name_mapping = {
        '3': "relu1_2",
        '8': "relu2_2",
        '15': "relu3_3",
        '22': "relu4_3"
    }

  def forward(self, x):
      output = {}
      for name, module in self.vgg_layers._modules.items():
        x = module(x)
        if name in self.layer_name_mapping:
          output[self.layer_name_mapping[name]] = x
      return LossOutput(**output)


vgg_model = vgg.vgg16(pretrained = True)
if torch.cuda.is_available():
  vgg_model.cuda()
loss_network = LossNetwork(vgg_model)
loss_network.eval()

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STYLE_IMAGE = "/content/drive/MyDrive/styles/udnie.jpg"
style_img = Image.open(STYLE_IMAGE).convert("RGB")
with torch.no_grad():
    style_img_tensor = transforms.Compose([
     transforms.ToTensor(),
     tensor_normalizer]
        )(style_img).unsqueeze(0)
    #assert np.sum(style_img - recover_image(style_img_tensor.numpy())[0].astype(np.uint8)) < 3 * style_img_tensor.size()[2] * style_img_tensor.size()[3]
    #style_img_tensor = style_img_tensor.cuda()

style_img_tensor.size()

import torch
import numpy as np
from PIL import Image

def recover_image(img_tensor):
    # Normalize tensor back to [0, 1] range and move to CPU
    img_tensor = img_tensor.to('cpu').detach()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
    img_tensor = img_tensor * std + mean
    img_tensor = torch.clamp(img_tensor, 0, 1)

    # Ensure img_tensor is in the expected format (H, W, C) for a single image
    if img_tensor.dim() == 4:
        # If it's a batch, select the first image, assuming you're debugging
        img_tensor = img_tensor[0]
    img_np = img_tensor.permute(1, 2, 0).numpy()

    # Convert numpy array to PIL Image
    img_np = (img_np * 255).astype(np.uint8)
    return Image.fromarray(img_np)

def save_debug_image(tensor_orig, tensor_transformed, filename):
    orig_img = recover_image(tensor_orig)
    transformed_img = recover_image(tensor_transformed)

    # Assuming both images are now PIL Images, combine them
    dst = Image.new('RGB', (orig_img.width + transformed_img.width, max(orig_img.height, transformed_img.height)))
    dst.paste(orig_img, (0, 0))
    dst.paste(transformed_img, (orig_img.width, 0))

    dst.save(filename)


# Move the tensor to CPU before passing it to recover_image
img_to_display = recover_image(style_img_tensor.squeeze(0).cpu())  # Remove batch dim and move to CPU
plt.imshow(img_to_display)


# Assuming loss_network is your model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_network = loss_network.to(device)  # Move the model to the appropriate device

# Ensure input tensors are on the same device as the model
style_img_tensor = style_img_tensor.to(device)

with torch.no_grad():
    style_loss_features = loss_network(style_img_tensor)
    gram_style = [gram_matrix(y).data for y in style_loss_features]


style_loss_features._fields

np.mean(gram_style[3].data.cpu().numpy())

np.mean(style_loss_features[3].data.cpu().numpy())

gram_style[0].numel()

'''
def save_debug_image(tensor_orig, tensor_transformed, filename):
    tensor_orig_np = recover_image(tensor_orig.detach())
    tensor_transformed_np = recover_image(tensor_transformed.detach())

    orig_img = Image.fromarray(tensor_orig_np)
    transformed_img = Image.fromarray(tensor_transformed_np)

    new_img = Image.new('RGB', (orig_img.width + transformed_img.width + 10, orig_img.height))
    new_img.paste(orig_img, (0, 0))
    new_img.paste(transformed_img, (orig_img.width + 10, 0))

    new_img.save(filename)
'''


%mkdir -p debug

transformer = TransformerNet()
mse_loss = torch.nn.MSELoss()
if torch.cuda.is_available():
  transformer.cuda()

!pip install comet_ml

from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

experiment = Experiment(
  api_key="t0zqSwN0yEbycxLxznUqLTkHU",
  project_name="style-transfer",
  workspace="gdi1u21",
  auto_metric_logging = False,
  auto_output_logging = False,
)

CONTENT_WEIGHT = 1
STYLE_WEIGHT = 1e5
LOG_INTERVAL = 200
REGULARIZATION = 1e-7

LR = 1e-3
optimizer = Adam(transformer.parameters(), LR)
transformer.train()
for epoch in range(10):
    loop=tqdm(train_loader)
    agg_content_loss = 0.
    agg_style_loss = 0.
    agg_reg_loss = 0.
    count = 0
    for x, _ in loop:
    #for batch_id, (x, _) in tqdm_notebook(enumerate(train_loader), total=len(train_loader)):
    #for batch_id, (x, _) in enumerate(train_loader):

        count +=1
        optimizer.zero_grad()

        x=x.cuda()
        y = transformer(x)

        with torch.no_grad():
                xc = x.detach()
        features_y = loss_network(y)
        features_xc = loss_network(xc)
        with torch.no_grad():
                f_xc_c = features_xc[2].detach()


        content_loss = CONTENT_WEIGHT * mse_loss(features_y[2], f_xc_c)

        reg_loss = REGULARIZATION * (
            torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) +
            torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])))

        style_loss = 0.
        # Compute style loss with normalized Gram matrices
        for m in range(len(features_y)):
            gram_s = gram_style[m]
            gram_y = gram_matrix(features_y[m])
            style_loss += STYLE_WEIGHT * mse_loss(gram_y, gram_s.expand_as(gram_y))


        total_loss = content_loss + style_loss + reg_loss
        total_loss.backward()
        optimizer.step()

        agg_content_loss += content_loss.item()
        agg_style_loss += style_loss.item()
        agg_reg_loss += reg_loss.item()
        #   loop.set_postfix(loss=agg_style_loss/count)
        if count % LOG_INTERVAL == 0:
            mesg = "{} [{}/{}] content: {:.6f}  style: {:.6f}  reg: {:.6f}  total: {:.6f}".format(
                        time.ctime(), count, len(train_dataset),
                        agg_content_loss / LOG_INTERVAL,
                        agg_style_loss / LOG_INTERVAL,
                        agg_reg_loss / LOG_INTERVAL,
                        (agg_content_loss + agg_style_loss + agg_reg_loss) / LOG_INTERVAL
                    )
            print(mesg)
            experiment.log_metric("content_loss", agg_content_loss / LOG_INTERVAL, epoch = epoch)
            experiment.log_metric("style_loss", agg_style_loss / LOG_INTERVAL)
            agg_content_loss = 0
            agg_style_loss = 0
            agg_reg_loss = 0
            transformer.eval()
            y = transformer(x)
            save_debug_image(x.data, y.data, "debug/{}_{}.png".format(epoch, count))
            transformer.train()


import glob
fnames = glob.glob(DATASET + r"/*/*")
len(fnames)

transformer = transformer.eval()

def recover_image(img_tensor):
    """
    Converts a tensor to a PIL Image, reversing the normalization.
    """
    if not isinstance(img_tensor, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor")

    img_tensor = img_tensor.to('cpu').detach()

    # Normalize tensor back to [0, 1] range
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
    img_tensor = img_tensor * std + mean
    img_tensor = torch.clamp(img_tensor, 0, 1)

    # Check if it's a single image (3D tensor) or a batch of images (4D tensor)
    # and remove the batch dimension if present
    if img_tensor.dim() == 4:
        img_tensor = img_tensor.squeeze(0)  # Remove batch dim assuming it's a single image in the batch

    # Ensure the tensor is 3D now
    if img_tensor.dim() != 3:
        raise ValueError("Expected input tensor to be 3D or 4D with a single image in the batch")

    # Convert tensor to numpy array
    img_np = img_tensor.permute(1, 2, 0).numpy()

    # Convert numpy array to PIL Image
    img_np = (img_np * 255).astype(np.uint8)
    return Image.fromarray(img_np)


img = Image.open(fnames[949]).convert('RGB')
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    tensor_normalizer  # Use the instance directly without calling it
])
img_tensor = transform(img).unsqueeze(0)
if torch.cuda.is_available():
    img_tensor = img_tensor.cuda()

img_output = transformer(img_tensor)  # Ensure img_tensor is a PyTorch tensor
plt.imshow(recover_image(img_tensor))  # This should work if img_tensor is a tensor

Image.fromarray(recover_image(img_output.data.cpu().numpy())[0])

save_model_path = "model_udnie.pth"
torch.save(transformer.state_dict(), save_model_path)

transformer.load_state_dict(torch.load(save_model_path))

# Assuming TransformerNet is already defined
# If the TransformerNet code is not already in your script, you need to define it here
model = TransformerNet()

# Load the state dictionary
state_dict = torch.load('/content/model_udnie.pth')

# Update the model instance with the loaded state dictionary
model.load_state_dict(state_dict)

# Set the model to evaluation mode if you are doing inference
model.eval()

# Now you can use the model for inference

import matplotlib.pyplot as plt

def plot_images(original, style, styled):
    """
    Plots original, style, and styled images side by side.

    Args:
    - original: The original image.
    - style: The style reference image.
    - styled: The result of applying style transfer to the original image.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ['Original Image', 'Style Image', 'Styled Image']
    images = [original, style, styled]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    plt.show()


import torch
import os
import random
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import glob

# Assuming TransformerNet and TensorNormalizer are already defined and initialized

# Load the TransformerNet model
model = TransformerNet()
model.load_state_dict(torch.load('/content/model_udnie.pth'))
model.eval()

# If you're using CUDA, move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define mean and std
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])

# Initialize the tensor normalizer
tensor_normalizer = TensorNormalizer(mean, std)

# Define the transformation
transform = transforms.Compose([
    transforms.Resize(256),  # Or your IMAGE_SIZE
    transforms.CenterCrop(256),  # Or your IMAGE_SIZE
    transforms.ToTensor(),
    tensor_normalizer
])

# Function to reverse normalization for visualization
def denormalize(tensor):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

# Load and transform the style image
style_img_path = STYLE_IMAGE  # Replace with your style image path
style_img = Image.open(style_img_path).convert('RGB')
style_tensor = transform(style_img).unsqueeze(0).to(device)

# Directory containing your COCO images
image_directory = '/content/drive/MyDrive/data_coco/train2017'

image_files = glob.glob(image_directory + '/*.jpg')

# Ensure we don't try to sample more images than available
num_samples = min(3, len(image_files))

if num_samples == 0:
    raise ValueError("No .jpg files found in the specified directory.")

selected_image_paths = random.sample(image_files, num_samples)  # Adjust number as needed based on availability

def apply_style_transfer(input_img_path):
    input_img = Image.open(input_img_path).convert('RGB')
    input_tensor = transform(input_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Reverse normalization for display
    output_tensor = denormalize(output_tensor)

    # Convert tensor to PIL Image for plotting
    styled_img_plt = transforms.ToPILImage()(output_tensor.cpu().squeeze(0))
    return styled_img_plt

for input_img_path in selected_image_paths:
    styled_img_plt = apply_style_transfer(input_img_path)

    # Convert tensors to PIL Images for plotting
    original_img_plt = Image.open(input_img_path)
    style_img_plt = Image.open(style_img_path)

    # Plot the images
    plot_images(original_img_plt, style_img_plt, styled_img_plt)


experiment.end()


from torch.optim import Adam
from torchvision import datasets
from torch.utils.data import DataLoader

# Assuming you have defined a loss function that combines content, style, and regularization losses
def compute_loss(model_output, content_targets, style_targets):
    # Implement your loss computation here
    return total_loss

# Prepare DataLoader for your training dataset
train_dataset = datasets.ImageFolder('/content/drive/MyDrive/data_coco', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Optimizer
optimizer = Adam(model.parameters(), lr=1e-3)

# Training Loop
model.train()
for epoch in range(epoch):
    for images, _ in train_loader:
        optimizer.zero_grad()
        if torch.cuda.is_available():
        # Forward pass
            outputs = model(images.cuda())

            # Compute loss
            loss = compute_loss(outputs, content_targets, style_targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")


