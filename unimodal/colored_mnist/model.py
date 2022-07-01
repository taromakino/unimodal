import torch
import torch.nn as nn
from torch.autograd import Variable

class SSVAE(nn.Module):
    def __init__(self, latent_dim):
        super(SSVAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.n_latents = latent_dim

    def sample_z(self, mu, logvar):
        if self.training:
            sd = torch.exp(logvar / 2) # Same as sqrt(exp(logvar))
            eps = torch.randn_like(sd)
            return mu + eps * sd
        else:
            return mu

    def forward(self, x, y):
        mu, logvar = self.encoder(x)
        z = self.sample_z(mu, logvar)
        img_reconst = self.decoder(z)
        return img_reconst, mu, logvar

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(784 * 3, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc31 = nn.Linear(512, latent_dim)
        self.fc32 = nn.Linear(512, latent_dim)
        self.swish = Swish()

    def forward(self, x):
        h = self.swish(self.fc1(x.view(-1, 784 * 3)))
        h = self.swish(self.fc2(h))
        return self.fc31(h), self.fc32(h)

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 784 * 3)
        self.swish = Swish()

    def forward(self, z):
        h = self.swish(self.fc1(z))
        h = self.swish(self.fc2(h))
        h = self.swish(self.fc3(h))
        return self.fc4(h)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)