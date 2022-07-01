from colored_mnist.model import SSVAE
from colored_mnist.data import make_data
from utils.ml import *
from utils.plotting import *

def plot_img(x_elem):
    plt.imshow(x_elem.reshape(28, 28, 3))

dpath = "results/spurious"
fpath = os.path.join(dpath, "optimal_weights.pt")

latent_dim = 256

x, y = make_data(True, 0)
x /= 255.
x, y = torch.tensor(x), torch.tensor(y[:, None])

model = SSVAE(latent_dim)
model.load_state_dict(torch.load(fpath, map_location="cpu"))

idx = 0
x_elem, y_elem = x[idx], y[idx]
x_reconst, mu, logvar = model(x_elem[None], y_elem[None])
x_reconst = torch.sigmoid(x_reconst).detach().numpy().squeeze()
plot_img(x_reconst)