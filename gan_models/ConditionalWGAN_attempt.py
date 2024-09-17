import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim, hidden_dim=64, image_channel=3):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # 1 * 1 * 100
            nn.ConvTranspose2d(in_channels=noise_dim, out_channels=hidden_dim * 8, kernel_size=4, stride=1),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.ReLU(inplace=True),
            # 4 * 4 * 512
            nn.ConvTranspose2d(in_channels=hidden_dim * 8, out_channels=hidden_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True),
            # 8 * 8 * 256
            nn.ConvTranspose2d(in_channels=hidden_dim * 4, out_channels=hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            # 16 * 16 * 128
            nn.ConvTranspose2d(in_channels=hidden_dim * 2, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            # 32 * 32 * 64
            nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=image_channel, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
            # 64 * 64 * 3
        )

    def forward(self, noise):
        noise = noise.view(noise.size(0), noise.size(1), 1, 1) # (batch_size, noise_dim, height=1, width=1)
        return self.gen(noise)


class Critic(nn.Module):
    def __init__(self, hidden_dim=64, image_channel=3):
        super(Critic, self).__init__()
        self.disc = nn.Sequential(
            # 64 * 64 * 3
            nn.Conv2d(in_channels=image_channel, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # 32 * 32 * 64
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # 16 * 16 * 128
            nn.Conv2d(in_channels=hidden_dim * 2, out_channels=hidden_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # 8 * 8 * 256
            nn.Conv2d(in_channels=hidden_dim * 4, out_channels=hidden_dim * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # 4 * 4 * 512
            nn.Conv2d(in_channels=hidden_dim * 8, out_channels=1, kernel_size=4, stride=1),
            # 1 * 1 * 1

        )

    def forward(self, image):
        image = self.disc(image)
        image = torch.flatten(image, 1)
        return image

input_size=28*28
output_size=10
hidden_size1=128
hidden_size2=64
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.flatten=nn.Flatten()
    self.layer1=nn.Linear(input_size,hidden_size1)
    self.layer2=nn.Linear(hidden_size1,hidden_size2)
    self.layer3=nn.Linear(hidden_size2,output_size)
    self.relu=nn.ReLU()
  def forward(self,x):
    x=self.flatten(x)
    x=self.layer1(x)
    x=self.relu(x)
    x=self.layer2(x)
    x=self.relu(x)
    x=self.layer3(x)

    return x

model=Net()
print(model)

def ReLU(x):
    if x >=0:
        return x
    return 0
def LReLU(x):
    if x>=0:
        return x
    return 0.01*x
x=[i for i in range(-10,10)]
u=[ReLU(z) for z in x]
v=[LReLU(z) for z in x]
plt.ylim(-0.1,2)
plt.plot(x,u,label='ReLU')
plt.plot(x,v,label='LReLU')
plt.legend()

model=model.cuda()
import torch.optim as optim
epochs=10
optimizer=optim.SGD(model.parameters(),lr=0.01)
loss_fn=nn.CrossEntropyLoss()
running_loss=0.0
for epoch in range(epochs):
  for i,data in enumerate(train_loader):
    optimizer.zero_grad()
    img,label=data
    img,label=img.cuda(),label.cuda()

    output=model(img)
    loss=loss_fn(output,label)

    loss.backward()
    optimizer.step()
    running_loss=0.99*running_loss+0.01*loss.item()
  print("loss {:.4f}".format(running_loss))

