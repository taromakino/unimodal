from arch.conv_vae import SSVAE
from colored_mnist.data import make_data
from utils.ml import *
from utils.plotting import *

def plot_img(x_elem):
    plt.imshow(x_elem.transpose(1, 2, 0))

dpath = "results/1,1/spurious"
fpath = os.path.join(dpath, "optimal_weights.pt")

p_flip_color = 0
sigma = 0.1
hidden_dims = [64, 128, 256]
latent_dim = 256

x0, x1, y = make_data(True, p_flip_color, sigma)
x0, x1, y = torch.tensor(x0), torch.tensor(x1[:, None]), torch.tensor(y[:, None])

x0_dim = x0.shape[1:]
x1_dim = x1.shape[1]
y_dim = y.shape[1]

model = SSVAE(x0_dim, x1_dim, hidden_dims, latent_dim, y_dim)
model.load_state_dict(torch.load(fpath, map_location="cpu"))

idx = 0
x0_elem, x1_elem, y_elem = x0[idx], x1[idx], y[idx]
x0_reconst, x1_mu, x1_logprec, mu, logvar = model(x0_elem[None], x1_elem[None], y_elem[None])
x0_reconst = torch.sigmoid(x0_reconst).detach().numpy().squeeze()
plot_img(x0_reconst)