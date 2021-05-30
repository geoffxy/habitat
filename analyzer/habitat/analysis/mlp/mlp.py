import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import random_split, DataLoader

import numpy as np

from tqdm import tqdm

from habitat.analysis.mlp.devices import get_device_features
from habitat.analysis.mlp.dataset import HabitatDataset


class MLPBase(nn.Module):
    def __init__(self, layers, layer_size):
        super().__init__()

        self.layers = nn.ModuleList()

        for idx in range(layers):
            self.layers.append(nn.Linear(layer_size, layer_size))
            self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class LinearMLP(nn.Module):
    def __init__(self, layers, layer_size):
        super().__init__()

        self.features = ["bias", "batch", "in_features", "out_features"]

        self.fc1 = nn.Linear(len(self.features) + 4, layer_size)
        self.mlp = MLPBase(layers, layer_size)
        self.fc2 = nn.Linear(layer_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.mlp(x)
        x = self.fc2(x)

        return x


class LSTMMLP(nn.Module):
    def __init__(self, layers, layer_size):
        super().__init__()

        self.features = ['bias', 'bidirectional', 'batch', 'seq_len', 'input_size', 'hidden_size', 'num_layers']

        self.fc1 = nn.Linear(len(self.features) + 4, layer_size)
        self.mlp = MLPBase(layers, layer_size)
        self.fc2 = nn.Linear(layer_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.mlp(x)
        x = self.fc2(x)

        return x


class Conv2DMLP(nn.Module):
    def __init__(self, layers, layer_size):
        super().__init__()

        self.features = ['bias', 'batch', 'image_size', 'in_channels', 'out_channels', 'kernel_size', 'stride',
                         'padding']

        # properly manage device parameters
        self.fc1 = nn.Linear(len(self.features) + 4, layer_size)
        self.mlp = MLPBase(layers, layer_size)
        self.fc2 = nn.Linear(layer_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.mlp(x)
        x = self.fc2(x)

        return x


class BMMMLP(nn.Module):
    def __init__(self, layers, layer_size):
        super().__init__()

        self.features = ["batch", "left", "middle", "right"]

        self.fc1 = nn.Linear(len(self.features) + 4, layer_size)
        self.mlp = MLPBase(layers, layer_size)
        self.fc2 = nn.Linear(layer_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.mlp(x)
        x = self.fc2(x)

        return x


class RuntimePredictor:
    def __init__(self, model_name, layers, layer_size, model_path=None):
        self.model_name = model_name
        self.layers = layers
        self.layer_size = layer_size

        self.model = {
            "linear": LinearMLP,
            "lstm": LSTMMLP,
            "conv2d": Conv2DMLP,
            "bmm": BMMMLP,
        }[self.model_name](layers, layer_size)

        self.device_params = ['mem', 'mem_bw', 'num_sm', 'single']

        self.mu = None
        self.sigma = None

        if model_path is not None:
            self.load_state(model_path)

    def load_state(self, path):
        checkpoint = torch.load(path)
        self.mu = checkpoint['mu']
        self.sigma = checkpoint['sigma']
        self.model.load_state_dict(checkpoint['model'])

    def save_state(self, path):
        checkpoint = {
            "mu": self.mu,
            "sigma": self.sigma,
            "model": self.model.state_dict()
        }

        torch.save(checkpoint, path)

    def _train(self):
        self.model.train()
        losses = []
        for batch_x, batch_y in tqdm(self.train_dataloader, leave=False, desc="Training"):
            batch_x = batch_x.float()
            batch_y = batch_y.float()

            self.optim.zero_grad()
            pred = self.model(batch_x.to(self.device))
            loss = self.criterion(pred, batch_y.reshape(pred.shape).to(self.device))

            losses.append(loss.item())

            loss.backward()
            self.optim.step()

        avg_loss = sum(losses) / len(losses)
        return avg_loss

    def _validate(self):
        eps = 1e-6

        self.model.eval()
        perc_errors = []
        for batch_x, batch_y in tqdm(self.val_dataloader, leave=False, desc="Validating"):
            batch_x = batch_x.float()
            batch_y = batch_y.float().numpy()

            pred = self.model(batch_x.to(self.device)).detach().cpu().numpy()
            pred = pred.reshape(batch_y.shape)

            perc_error = np.divide(np.abs(pred - batch_y) + eps, batch_y + eps)
            perc_errors.append(perc_error)

        perc_errors_np = np.concatenate(perc_errors)
        mean_perc_err = float(np.mean(perc_errors_np))
        max_perc_err = np.amax(perc_errors_np)

        return mean_perc_err, max_perc_err

    def train_with_dataset(self, dataset_path, epochs=40, use_cuda=True):
        from torch.utils.tensorboard import SummaryWriter

        # declare device
        self.device = torch.device("cuda") if use_cuda else torch.device("cpu")
        self.model = self.model.to(self.device)

        # construct dataset loaders
        self.dataset = HabitatDataset(dataset_path, self.model.features)

        # get normalization parameters from dataset loader
        self.mu, self.sigma = self.dataset.mu, self.dataset.sigma

        # train/val split
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        train, val = random_split(self.dataset, (train_size, val_size))

        self.train_dataloader = DataLoader(train, batch_size=512, shuffle=True)
        self.val_dataloader = DataLoader(val, batch_size=512, shuffle=False)

        # implement losses and optimizers
        def MAPELoss(output, target):
            return torch.mean(torch.abs((target - output) / target))

        self.criterion = MAPELoss
        self.optim = torch.optim.Adam(self.model.parameters(), lr=5e-4, weight_decay=1e-4)

        # set up tensorboard logging
        self.writer = SummaryWriter("logs/%s_%d_%d_%d" % (
            self.model_name, self.layers, self.layer_size, int(time.time())
        ))

        # run training loops
        min_perc_err = 1e9
        for epoch in range(epochs):
            # change learning rate
            if epoch == epochs // 2:
                print("Epoch %d: Changing learning rate to 1e-4" % epoch)
                for g in self.optim.param_groups:
                    # g['lr'] = 5e-5
                    g['lr'] = 1e-4

            # train
            train_loss = self._train()
            print("epoch %3s\tloss: %.4f" % (str(epoch), train_loss), end="\t")
            self.writer.add_scalar("train_loss", train_loss, epoch)

            # validate
            avg_err, max_err = self._validate()
            print("val avg: %.4f, max: %.4f" % (avg_err, max_err), end="\t")
            self.writer.add_scalar("validation avg_err", avg_err, epoch)
            self.writer.add_scalar("validation max_err", max_err, epoch)

            # save model if good
            if avg_err < min_perc_err:
                min_perc_err = avg_err
                self.save_state("saved_models/%s/model.pth" % self.model_name)
                print("\t(new best, saving!)")
            else:
                print()

        self.writer.close()

    def predict(self, kernel_arguments, device_name):
        # move to CPU and change to single prec
        self.model = self.model.to(torch.device('cpu')).float()

        device_features = get_device_features(device_name, self.device_params)
        kernel_params = kernel_arguments
        features = np.array(kernel_params + device_features)

        # normalize features
        features = (features - self.mu) / self.sigma

        # predict runtime with model
        pred = self.model(torch.from_numpy(features).float())
        pred = float(pred.squeeze().item())

        return pred
