import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from Config import DEVICE, example_save_dir, checkpoints_dir, plot_save_dir, train_path, batch_size, val_path, learning_rate, num_epochs, save_interval
from Dataset import DemosaicDataset
from Eval import eval_model
from Model import DemosaicModel


def plot_losses(train_losses, val_losses, epoch, save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure()
    plt.plot(np.array(range(1, len(train_losses) + 1)), train_losses, label='Training Loss')
    plt.plot(np.array(range(1, len(val_losses) + 1)), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.legend()
    plt.ylim((0, np.quantile(train_losses, 0.9)))
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'loss_epoch_{epoch}.png'))
    plt.close()


def calc_loss(x, y):
    xc = x[:, :, 8:-8, 8:-8]
    yc = y[:, :, 8:-8, 8:-8]
    xdh = xc[:, :, :, 1:] - xc[:, :, :, :-1]
    xdv = xc[:, :, 1:, :] - xc[:, :, :-1, :]
    ydh = yc[:, :, :, 1:] - yc[:, :, :, :-1]
    ydv = yc[:, :, 1:, :] - yc[:, :, :-1, :]
    l2 = 10000 * torch.nn.functional.mse_loss(xc, yc)
    ldh = 10000 * torch.nn.functional.mse_loss(xdh, ydh)
    ldv = 10000 * torch.nn.functional.mse_loss(xdv, ydv)
    return l2 + ldh + ldv


def train():
    os.makedirs(example_save_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(plot_save_dir, exist_ok=True)

    train_crop_transform = v2.Compose([
        v2.RandomCrop((256, 256)),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.ToDtype(dtype=torch.float32, scale=True),
    ])

    train_augment_transform = v2.Compose([
        v2.ColorJitter(),
        v2.RandomChannelPermutation()
    ])

    train_dataset = DemosaicDataset(root_dir=train_path, len_factor=50, crop_transform=train_crop_transform, augment_transform=train_augment_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    val_dataset = DemosaicDataset(root_dir=val_path, len_factor=1, crop_transform=v2.CenterCrop((512, 512)))
    val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, num_workers=4, pin_memory=False)

    # leave the output at gamma=2, because the loss function also operates at gamma=2
    model = DemosaicModel(gamma2=True).to(DEVICE)

    train_losses_epoch = []
    val_losses_epoch = []
    iteration = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
        model.train()
        epoch_train_loss = 0.0
        num_batches = 0

        for batch_idx, (bayer, target) in enumerate(train_dataloader):
            bayer = bayer.to(DEVICE)
            target = torch.sqrt_(target.to(DEVICE))

            pred = model(bayer)
            loss = calc_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            num_batches += 1

            if iteration % 10 == 0:
                print(f"Iteration [{iteration}], Loss: {loss.item():.4f}")

            if iteration % save_interval == 0:
                torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}, os.path.join(checkpoints_dir, f'model_iter_{iteration}.pth'))
                print(f"Saved model checkpoint at iteration {iteration}")

                avg_train_loss = epoch_train_loss / save_interval
                train_losses_epoch.append(avg_train_loss)
                epoch_train_loss = 0.0

                model.eval()
                avg_val_loss = eval_model(val_dataloader, model, calc_loss, iteration, example_save_dir)
                val_losses_epoch.append(avg_val_loss)
                print(f"Validation Loss after iteration {iteration}: {avg_val_loss:.4f}")
                model.train()

            iteration += 1
        plot_losses(train_losses_epoch, val_losses_epoch, epoch + 1, save_dir=plot_save_dir)


if __name__ == '__main__':
    train()
