import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


class GarmentClassifier(nn.Module):
    def __init__(self):
        super(GarmentClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def train_one_epoch(model, loss_fn, training_loader, epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    loss_fn = torch.nn.CrossEntropyLoss()
    dummy_outputs = torch.rand(4, 10)
    dummy_labels = torch.tensor([1, 5, 3, 7])
    loss = loss_fn(dummy_outputs, dummy_labels)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for i, data in enumerate(training_loader):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()

        running_loss += loss.item()

        if i % 1000 == 999:
            last_loss = running_loss / 1000
            print(f'batch: {i+1} loss:{last_loss}')
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0

    return last_loss


def main():

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    training_set = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
    validation_set = torchvision.datasets.FashionMNIST('./data', train=False, )

    training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=4, shuffle=False)

    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

    dataiter = iter(training_loader)
    images, labels = next(dataiter)

    img_grid = torchvision.utils.make_grid(images)
    matplotlib_imshow(img_grid, one_channel=True)
    print(' '.join(classes[labels[j]] for j in range(4)))

    print(f'Training set has {len(training_set)} instances')
    print(f'Validation set has {len(validation_set)} instances')

    model = GarmentClassifier()
    loss_fn = torch.nn.CrossEntropyLoss()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/fashion_trainer_{timestamp}')
    epoch_number = 0

    EPOCHS = 5

    best_vloss = 1_000_000

    for epoch in range(EPOCHS):
        print(f'EPOCH: {epoch_number + 1}')
        model.train(True)
        avg_loss = train_one_epoch(model, loss_fn, training_loader, epoch_number, writer)

        running_vlos = 0.0

        model.eval()

        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vlos += vloss
        avg_loss = running_vlos / (i+1)

        # Untill here!


    print('test')


if __name__ == "__main__":
    main()