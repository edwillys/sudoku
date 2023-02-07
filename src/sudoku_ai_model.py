import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import logging
from pathlib import Path
import time
import datetime as dt
import random


class ConvReluMaxPool2D(nn.Sequential):
    def __init__(self, numch_in, numch_out, kernel_size,
                 stride, padding, maxpool_kernel_size) -> None:
        super().__init__(
            nn.Conv2d(numch_in, numch_out, kernel_size, stride, padding),
            nn.MaxPool2d(maxpool_kernel_size)
            nn.ReLU(),
        )


class Model(nn.Module):
    def __init__(self, numch_out: int = 47, h_in: int = 28, w_in: int = 28) -> None:
        super().__init__()

        stride = 1
        dilation = 1
        padding = 2
        conv_ksize = 3
        maxpool_ksize = 2
        # 1 at the beginning because it is a gray scale image as an input
        # 16 at the end because we're interested on the 10 digits + the letters from A to F
        numch = [1, 32, 32]
        

        h = h_in
        w = h_in
        for _ in range(len(numch) - 1):
            h = int((h + 2 * padding - dilation * (conv_ksize - 1) - 1) / stride + 1)
            w = int((w + 2 * padding - dilation * (conv_ksize - 1) - 1) / stride + 1)
            h = h // maxpool_ksize
            w = w // maxpool_ksize


        self.conv = [
            ConvReluMaxPool2D(numch[i], numch[i+1],
                              conv_ksize, stride, padding, maxpool_ksize)
            for i in range(len(numch) - 1)
        ]
        # TODO: magic number 7
        self.fully_connected = nn.Linear(numch[-1] * h * w, numch_out)

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        x = x.view(x.size(0), -1)
        return self.fully_connected(x)


class Trainer():
    def __init__(self, model: nn.Module, loader_train: DataLoader, loader_test: DataLoader, epochs: int = 100) -> None:
        self.epochs = epochs
        self.device = "cuda" if torch.has_cuda else "cpu"
        self.model = model
        self.model.to(self.device)
        self.fn_loss = nn.CrossEntropyLoss()
        self.fn_optimizer = optim.SGD(model.parameters(), lr=0.01)
        self.loader_train = loader_train
        self.loader_test = loader_test

    def train(self):
        batch_len = len(self.loader_train)
        # set to train mode
        self.model.train()
        for epoch in range(self.epochs):
            for i, (X_train, y_train) in enumerate(self.loader_train):
                # forward pass
                y_logits = self.model(X_train.to(self.device))
                # calculate the loss
                loss = self.fn_loss(y_logits, y_train.to(self.device))
                # reset gradient
                self.fn_optimizer.zero_grad()
                # backward propagation
                loss.backward()
                self.fn_optimizer.step()

                if (i % (batch_len // 10)) == 0:
                    logging.info(
                        f"Epoch {epoch}/{self.epochs} | Batch {i}/{batch_len} | Loss {loss: .5f}")

    def test(self):
        self.model.eval()
        test_len = len(self.loader_test)
        accuracy = np.zeros(test_len)
        with torch.inference_mode():
            for i, (X_test, y_test) in enumerate(self.loader_test):
                test_logits = self.model(X_test.to(self.device))
                pred_y = torch.argmax(test_logits, 1)
                accuracy[i] = (pred_y == y_test).sum().item() / \
                    float(y_test.size(0))
                if (i % (test_len // 10)) == 0:
                    logging.info(
                        f"Test Accuracy[{i}] = {accuracy[i]:.2f}")

        logging.info(f"Test Average Accuracy = {np.average(accuracy):.2f}")

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("Preparing model and dataset")
    model = Model()
    base_path = Path(__file__).parent / Path("../res/DL")
    base_path.mkdir(parents=True, exist_ok=True)
    data_train = datasets.EMNIST(
        str(base_path), "balanced", train=True, download=True, transform=ToTensor())
    data_test = datasets.EMNIST(
        str(base_path), "balanced", train=False, download=True, transform=ToTensor())
    # loaders
    batch_size = 100
    num_workers = 8
    torch.manual_seed(666)
    g = torch.Generator()
    g.manual_seed(666)
    loader_train = DataLoader(
        data_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
    loader_test = DataLoader(
        data_test, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, worker_init_fn=seed_worker, generator=g)

    trainer = Trainer(model, loader_train, loader_test)
    logging.info("Beggining to train the model")
    t_start = time.time()
    trainer.train()
    trainer.save(str(base_path / "model_0.pth"))
    t_end = time.time()
    logging.info(
        f"Finished the training in {dt.timedelta(seconds=(t_end - t_start))}")
    trainer.test()


if __name__ == "__main__":
    main()
