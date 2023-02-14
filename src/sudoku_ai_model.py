import torch
import torch_directml
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision import transforms
import torchvision
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
            nn.MaxPool2d(maxpool_kernel_size),
            nn.ReLU()
        )


class ConvRelu(nn.Sequential):
    def __init__(self, numch_in, numch_out, kernel_size,
                 stride, padding) -> None:
        self.nodes = [
            nn.Conv2d(numch_in, numch_out, kernel_size, stride, padding),
            nn.ReLU(),
            nn.BatchNorm2d(numch_out)
        ]
        super().__init__(*self.nodes)


class ConvDropout(nn.Sequential):
    def __init__(self, numch: list[int], ksizes: list[int], strides: list[int], paddings: list[int],
                 p_dropout: float) -> None:
        self.nodes = [
            ConvRelu(numch[i], numch[i+1],
                     ksizes[i], strides[i], paddings[i])
            for i in range(len(numch) - 1)
        ]
        self.nodes += [
            nn.Dropout2d(p_dropout)
        ]
        super().__init__(*self.nodes)


class MyLeNet5(nn.Sequential):
    def __init__(self, numch_in=1, numch_out=10, numch_conv=[6, 16], numch_dense=[120, 84],
                 shape_in: tuple[int, int] = (32, 32), dropout_rate=0.3, transf=None):
        numch_conv = [numch_in, *numch_conv]
        ksize_conv2d = 5
        stride_conv2d = 1
        ksize_maxpool = 2
        stride_maxpool = 2
        conv_blocks = [
            nn.Sequential(
                nn.Conv2d(numch_conv[i], numch_conv[i+1],
                          kernel_size=ksize_conv2d, stride=stride_conv2d),
                nn.BatchNorm2d(numch_conv[i+1]),
                nn.MaxPool2d(kernel_size=ksize_maxpool, stride=stride_maxpool),
                nn.ReLU(),
                nn.Dropout2d(dropout_rate),
            )
            for i in range(len(numch_conv) - 1)
        ]
        conv = nn.Sequential(*conv_blocks)

        self.shape_in = shape_in
        h, w = shape_in
        dilation = 1
        padding = 0
        for _ in range(len(numch_conv) - 1):
            h = int((h + 2 * padding - dilation *
                    (ksize_conv2d - 1) - 1) / stride_conv2d + 1)
            w = int((w + 2 * padding - dilation *
                    (ksize_conv2d - 1) - 1) / stride_conv2d + 1)
            h //= ksize_maxpool
            w //= ksize_maxpool
            h = max(h, 1)
            w = max(w, 1)

        numch_dense = [numch_conv[-1] * h * w, *numch_dense]

        dense_blocks = [
            nn.Sequential(
                nn.Linear(numch_dense[i], numch_dense[i+1]),
                nn.ReLU(),
            )
            for i in range(len(numch_dense) - 1)
        ]
        dense_blocks += [
            nn.Linear(numch_dense[-1], numch_out),
        ]
        dense = nn.Sequential(*dense_blocks)
        super().__init__(conv, dense)
        self.conv = conv
        self.dense = dense
        self.transf = transf

    def forward(self, x):
        if self.transf:
            x = self.transf(x)
        out = self.conv(x)
        out = out.flatten(1, -1)
        return self.dense(out)


class Model(nn.Module):
    def __init__(self, numch_out: int = 47, h_in: int = 32, w_in: int = 32) -> None:
        super().__init__()

        strides = [
            [1, 1, 2],
            [1, 1, 2],
            [1],
        ]
        kernels = [
            [3, 3, 5],
            [3, 3, 5],
            [4],
        ]
        paddings = [
            ["same", "same", 1],
            ["same", "same", 1],
            [1],
        ]
        numch = [
            [1, 32, 32, 32],
            [32, 64, 64, 64],
            [64, 128],
        ]

        self.conv = [
            ConvDropout(c, k, s, p, 0.2)
            for c, k, s, p in zip(numch, kernels, strides, paddings)
        ]

        h = h_in
        w = w_in
        dilation = 1
        for ks, ss, ps in zip(kernels, strides, paddings):
            for k, s, p in zip(ks, ss, ps):
                if p != "same":
                    if p == "valid":
                        p = 0
                    h = int((h + 2 * p - dilation * (k - 1) - 1) / s + 1)
                    w = int((w + 2 * p - dilation * (k - 1) - 1) / s + 1)
                    h = max(h, 1)
                    w = max(w, 1)

        numch_dense = [
            numch[-1][-1] * h * w,
            64
        ]

        dense_blocks = [
            nn.Sequential(
                nn.Linear(numch_dense[i], numch_dense[i+1]),
                nn.ReLU(),
            )
            for i in range(len(numch_dense) - 1)
        ]
        dense_blocks += [
            nn.Linear(numch_dense[-1], numch_out),
        ]

        self.fully_connected = nn.Sequential(*dense_blocks)

    def to(self, dtype):
        super().to(dtype)
        for conv in self.conv:
            conv.to(dtype)
        self.fully_connected.to(dtype)

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        x = x.flatten(1, -1)
        x = self.fully_connected(x)
        return x

# class Model(nn.Module):
#    def __init__(self, numch_out: int = 47, h_in: int = 28, w_in: int = 28) -> None:
#        super().__init__()
#
#        stride = 1
#        dilation = 1
#        padding = 2
#        conv_ksize = 3
#        maxpool_ksize = 2
#        # 1 at the beginning because it is a gray scale image as an input
#        # 16 at the end because we're interested on the 10 digits + the letters from A to F
#        numch = [1, 32, 32]
#
#        h = h_in
#        w = h_in
#        for _ in range(len(numch) - 1):
#            h = int((h + 2 * padding - dilation * (conv_ksize - 1) - 1) / stride + 1)
#            w = int((w + 2 * padding - dilation * (conv_ksize - 1) - 1) / stride + 1)
#            h = h // maxpool_ksize
#            w = w // maxpool_ksize
#
#
#        self.conv = [
#            ConvReluMaxPool2D(numch[i], numch[i+1],
#                              conv_ksize, stride, padding, maxpool_ksize)
#            for i in range(len(numch) - 1)
#        ]
#        # TODO: magic number 7
#        self.fully_connected = nn.Linear(numch[-1] * h * w, numch_out)
#
#    def forward(self, x):
#        for conv in self.conv:
#            x = conv(x)
#        x = x.view(x.size(0), -1)
#        return self.fully_connected(x)


class Trainer():
    def __init__(self, model: nn.Module, loader_train: DataLoader, loader_test: DataLoader,
                 epochs: int = 40, test_in_train: bool = False) -> None:
        self.epochs = epochs
        # self.device = "cuda" if torch.has_cuda else "cpu"
        self.device = torch_directml.device()
        self.model = model
        self.model.to(self.device)
        self.fn_loss = nn.CrossEntropyLoss()
        self.fn_optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.loader_train = loader_train
        self.loader_test = loader_test
        self.avg_alpha = 0.99
        self.test_in_train = test_in_train

    def train(self):
        batch_len = len(self.loader_train)
        # set to train mode
        self.model.train()
        # just to get rid of annoying pylint warning
        acc_stats = (0, 0, 0)
        loss_stats = (0, 0, 0)
        for epoch in range(self.epochs):
            stats_init = False
            rotations = [0., -1, 1, -2., 2., -4, 4]
            for rotation in rotations:
                for i, (X_train, y_train) in enumerate(self.loader_train):
                    if rotation != 0:
                        X_train = transforms.functional.rotate(
                            X_train, rotation)
                    X_train = X_train.to(self.device)
                    y_train = y_train.to(self.device)
                    # forward pass
                    y_logits = self.model(X_train)
                    # calculate the loss
                    loss = self.fn_loss(y_logits, y_train)
                    # reset gradient
                    self.fn_optimizer.zero_grad()
                    # backward propagation
                    loss.backward()
                    self.fn_optimizer.step()
                    acc = self.accuracy(y_logits, y_train)
                    # stats
                    if not stats_init:
                        acc_stats = (acc, acc, acc)  # avg, min, max
                        loss_stats = (loss, loss, loss)  # avg, min, max
                        stats_init = True
                    acc_stats = self.stats(acc, *acc_stats)
                    loss_stats = self.stats(loss, *loss_stats)

                    if i == (batch_len // 2) or i == (batch_len - 1):
                        str_log = (f"Epoch {epoch}/{self.epochs} | Batch {i}/{batch_len} | Rotation {rotation}\n"
                                   f"  (Train) Accuracy: {acc_stats[0]:.3f} / {acc_stats[1]:.3f} / {acc_stats[2]:.3f} | "
                                   f"Loss {loss_stats[0]:.3f} / {loss_stats[1]:.3f} / {loss_stats[2]:.3f}\n"
                                   )
                        if self.test_in_train:
                            acc_stats_test, loss_stats_test = self.test()
                            self.model.train()  # need to set the model to test again
                            str_log += (
                                f"  (Test ) Accuracy: {acc_stats_test[0]:.3f} / {acc_stats_test[1]:.3f} / {acc_stats_test[2]:.3f} | "
                                f"Loss {loss_stats_test[0]:.3f} / {loss_stats_test[1]:.3f} / {loss_stats_test[2]:.3f}"
                            )
                        logging.info(str_log)

    def stats(self, x, y_1, val_min, val_max):
        val_min = min(val_min, x)
        val_max = max(val_max, x)
        val_avg = self.avg_alpha * y_1 + (1.0 - self.avg_alpha) * x

        return (val_avg, val_min, val_max)

    def accuracy(self, y_logits, y_ref):
        pred_y = torch.argmax(y_logits, 1)
        acc = (pred_y == y_ref).sum().item() / \
            float(y_ref.size(0))
        return acc

    def test(self):
        self.model.eval()
        self.model.to("cpu")  # TODO: why do we need this?
        stats_init = False
        # just to get rid of annoying pylint warning
        acc_stats = (0, 0, 0)
        loss_stats = (0, 0, 0)
        with torch.inference_mode():
            for X_test, y_test in self.loader_test:
                # X_test = X_test.to(self.device)
                # y_test = y_test.to(self.device)
                test_logits = self.model(X_test)
                loss = self.fn_loss(test_logits, y_test)
                acc = self.accuracy(test_logits, y_test)
                if not stats_init:
                    acc_stats = (acc, acc, acc)  # avg, min, max
                    loss_stats = (loss, loss, loss)  # avg, min, max
                    stats_init = True
                acc_stats = self.stats(acc, *acc_stats)
                loss_stats = self.stats(loss, *loss_stats)
        self.model.to(self.device)
        return acc_stats, loss_stats

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main(should_train=True, should_test=True, fpath_load=None, fpath_save=None):
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logging.info("Preparing model and dataset")

    base_path = Path(__file__).parent / Path("../res/DL")
    base_path.mkdir(parents=True, exist_ok=True)

    shape = (32, 32)
    data_train = datasets.EMNIST(
        str(base_path), "byclass", train=True, download=True,
        transform=transforms.Compose([
            transforms.Resize(shape),
            transforms.RandomHorizontalFlip(1.0),
            transforms.RandomRotation((90, 90)),
            transforms.ToTensor()
        ])
    )
    data_test = datasets.EMNIST(
        str(base_path), "byclass", train=False, download=True,
        transform=transforms.Compose([
            transforms.Resize(shape),
            transforms.RandomHorizontalFlip(1.0),
            transforms.RandomRotation((90, 90)),
            transforms.ToTensor()
        ])
    )
    # We're only interested in the hexadecimal characters. However, we don't use the balanced
    # set from EMNIST because if we balance ourselves with our wanted subset, we get more training
    # data
    num_classes = 16  # 0..1..A..F, excluding the remaining letters
    cnt_min = 2 ** 31
    indices = []
    for i in range(num_classes):
        idx = torch.argwhere((data_train.targets == i)).flatten().numpy()
        indices += [idx]
        cnt_min = min(cnt_min, len(idx))

    for i in range(num_classes):
        indices[i] = indices[i][:cnt_min]
    
    indices = np.array(indices).flatten()
    indices.sort()

    # We're only interested in the hexadecimal characters
    data_train = Subset(data_train, indices)
    # The testing data doesn't need to be balanced
    data_test = Subset(data_test, torch.argwhere(
        data_test.targets < num_classes).flatten().numpy())

    batch_size = 64
    num_workers = 1
    torch.manual_seed(666)
    g = torch.Generator()
    g.manual_seed(666)
    loader_train = DataLoader(
        data_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
    loader_test = DataLoader(
        data_test, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, worker_init_fn=seed_worker, generator=g)

    # model = Model(numch_out=10)
    # model = Model()
    model = MyLeNet5(
        numch_out=num_classes, shape_in=shape, numch_conv=[32, 64],
        transf=transforms.Normalize(0.1307, 0.3081)
    )
    if fpath_load:
        model.load_state_dict(torch.load(base_path / Path(fpath_load)))

    trainer = Trainer(model, loader_train, loader_test, epochs=5)

    if should_train:
        logging.info("Beginning the training")
        t_start = time.time()
        trainer.train()
        if fpath_save:
            trainer.save(str(base_path / Path(fpath_save)))
        logging.info(
            f"Finished the training in {dt.timedelta(seconds=(time.time() - t_start))}")
    if should_test:
        logging.info("Beginning the test")
        t_start = time.time()
        acc_stats_test, loss_stats_test = trainer.test()
        logging.info((
            f"Test Accuracy: {acc_stats_test[0]:.3f} / {acc_stats_test[1]:.3f} / {acc_stats_test[2]:.3f}\n"
            f"Test Loss: {loss_stats_test[0]:.3f} / {loss_stats_test[1]:.3f} / {loss_stats_test[2]:.3f}"
        ))
        logging.info(
            f"Finished the test in {dt.timedelta(seconds=(time.time() - t_start))}")


if __name__ == "__main__":
    # main(fpath_load="model_0.pth", should_train=False)
    main(fpath_save="model_1.pth")
