from functools import partial
from utils.ml import *
from torch.optim import Adam
from colored_mnist.data import make_data
from model import SSVAE
from argparse import ArgumentParser

def split_data(x, y, trainval_ratios):
    assert sum(trainval_ratios) == 1
    x /= 255.
    n_train, n_val = [int(len(x) * split_ratio) for split_ratio in trainval_ratios]
    x_train, y_train = x[:n_train], y[:n_train]
    x_val, y_val = x[n_train:n_train + n_val], y[n_train:n_train + n_val]
    if sum(trainval_ratios) == 1:
        return (x_train, y_train), (x_val, y_val)
    else:
        x_test, y_test = x[n_train + n_val:], y[n_train + n_val:]
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def main(args):
    set_seed(args.seed)

    x_trainval_det, y_trainval_det = make_data(True, 0)
    x_trainval_nondet, y_trainval_nondet = make_data(True, args.p_flip_color)

    x_test_det, y_test_det = make_data(False, 0)
    x_test_nondet, y_test_nondet = make_data(False, args.p_flip_color)

    (x_train_det, y_train_det), (x_val_det, y_val_det) = \
        split_data(x_trainval_det, y_trainval_det, args.trainval_ratios)
    (x_train_nondet, y_train_nondet), (x_val_nondet, y_val_nondet) = \
        split_data(x_trainval_nondet, y_trainval_nondet, args.trainval_ratios)

    x_train_det, x_val_det, x_test_det = torch.tensor(x_train_det), torch.tensor(x_val_det), torch.tensor(x_test_det)
    y_train_det, y_val_det, y_test_det = torch.tensor(y_train_det)[:, None], torch.tensor(y_val_det)[:, None], \
        torch.tensor(y_test_det)[:, None]

    x_train_nondet, x_val_nondet, x_test_nondet = torch.tensor(x_train_nondet), torch.tensor(x_val_nondet), \
        torch.tensor(x_test_nondet)
    y_train_nondet, y_val_nondet, y_test_nondet = torch.tensor(y_train_nondet)[:, None], torch.tensor(y_val_nondet)[:,None], \
        torch.tensor(y_test_nondet)[:, None]

    x_train_union, y_train_union = torch.vstack((x_train_det, x_train_nondet)), torch.vstack((y_train_det, y_train_nondet))
    x_val_union, y_val_union = torch.vstack((x_val_det, x_val_nondet)), torch.vstack((y_val_det, y_val_nondet))
    x_test_union, y_test_union = torch.vstack((x_test_det, x_test_nondet)), torch.vstack((y_test_det, y_test_nondet))

    data_det = make_dataloaders((x_train_det, y_train_det), (x_val_det, y_val_det), (x_test_det, y_test_det), args.batch_size)
    data_union = make_dataloaders((x_train_union, y_train_union), (x_val_union, y_val_union), (x_test_union, y_test_union),
        args.batch_size)

    train_f = partial(train_epoch_vae, n_anneal_epochs=args.n_anneal_epochs)

    model_det = SSVAE(args.latent_dim)
    model_union = SSVAE(args.latent_dim)
    model_det.to(make_device())
    model_union.to(make_device())
    optimizer_det = Adam(model_det.parameters(), lr=args.lr)
    optimizer_union = Adam(model_union.parameters(), lr=args.lr)

    dpath_spurious = os.path.join(args.dpath, "spurious")
    dpath_union = os.path.join(args.dpath, "union")
    os.makedirs(dpath_spurious, exist_ok=True)
    os.makedirs(dpath_union, exist_ok=True)

    train_eval_loop(*data_det, model_det, optimizer_det, train_f, eval_epoch_vae, dpath_spurious, args.n_epochs)
    train_eval_loop(*data_union, model_union, optimizer_union, train_f, eval_epoch_vae, dpath_union, args.n_epochs)

    device = make_device()
    kldivs_det, kldivs_union = [], []
    data_test_union = data_union[-1]
    model_det.eval()
    model_union.eval()
    for x_batch, y_batch in data_test_union:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        kldivs_det.append(posterior_kldiv(*model_det.posterior_params(x_batch, y_batch)).item())
        kldivs_union.append(posterior_kldiv(*model_union.posterior_params(x_batch, y_batch)).item())
    print(f"det={np.mean(kldivs_det):.3f}, union={np.mean(kldivs_union):.3f}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dpath", type=str, default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--p-flip-color", type=float, default=0.5)
    parser.add_argument("--trainval-ratios", nargs="+", type=float, default=[0.8, 0.2])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--n-anneal-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--latent-dim", type=int, default=256)
    main(parser.parse_args())