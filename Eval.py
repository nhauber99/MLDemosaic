import os

import torch
from torch.utils.data import DataLoader

from Config import DEVICE, example_save_dir
from Dataset import DemosaicDataset
from IPPModel import IPPModel, DemosaicMethod
from Model import DemosaicModel, BilinearModel
from torchvision.transforms import v2
from torchvision import utils


def save_example_images(bayer, pred, target, name, save_dir='examples'):
    os.makedirs(save_dir, exist_ok=True)
    bayer_rgb = bayer.repeat(1, 3, 1, 1)
    bayer_rgb = torch.clamp(bayer_rgb, 0, 1)
    pred = torch.clamp(pred, 0, 1)
    target = torch.clamp(target, 0, 1)
    concatenated = torch.cat((bayer_rgb, pred, target, 10 * torch.abs(pred - target)), dim=2)
    concatenated = torch.unbind(concatenated, dim=0)
    row1 = torch.cat(concatenated[:len(concatenated) // 2], dim=2)
    row2 = torch.cat(concatenated[len(concatenated) // 2:], dim=2)
    concatenated = torch.cat([row1, row2], dim=1)
    utils.save_image(concatenated ** 0.4545, os.path.join(save_dir, f'example_{name}.png'))


def calculate_psnr(pred, target, max_pixel_value=1.0):
    mse = torch.nn.functional.mse_loss(pred[:, :, 8:-8, 8:-8], target[:, :, 8:-8, 8:-8])
    return 20 * torch.log10(max_pixel_value / torch.sqrt(mse))


def eval_model(val_dataloader, model, loss_fn, name, example_save_dir):
    val_loss = 0.0
    val_batches = 0
    with torch.no_grad():
        for val_bayer, val_target in val_dataloader:
            val_bayer = val_bayer.to(DEVICE)
            val_target = val_target.to(DEVICE)
            if model.gamma2:
                val_target = torch.sqrt_(val_target)
            val_pred = model.forward(val_bayer)

            v_loss = loss_fn(val_pred, val_target)
            val_loss += v_loss.item()
            val_batches += 1
            if model.gamma2:
                val_target = torch.square_(val_target)
                val_pred = torch.square_(val_pred)
            save_example_images(val_bayer, val_pred, val_target, name, save_dir=example_save_dir)
    avg_val_loss = val_loss / val_batches
    return avg_val_loss


def eval(name):
    model = DemosaicModel().to(DEVICE)
    model.load_state_dict(torch.load('model.pth', map_location=DEVICE))

    val_dataset = DemosaicDataset(root_dir=os.path.join('D:', name), len_factor=1, transform=v2.CenterCrop((512, 512)))
    val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, num_workers=4, pin_memory=False)

    bilinear_model = BilinearModel().to(DEVICE)
    ahd_model = IPPModel(DemosaicMethod.AHD).to(DEVICE)
    vng_model = IPPModel(DemosaicMethod.VNG).to(DEVICE)

    model.eval()
    print(f'PSNR ({name}):')
    print(f"  Bilinear: {eval_model(val_dataloader, bilinear_model, calculate_psnr, f'{name}_bilinear', example_save_dir):.2f}")
    print(f"  AHD: {eval_model(val_dataloader, ahd_model, calculate_psnr, f'{name}_ahd', example_save_dir):.2f}")
    print(f"  VNG: {eval_model(val_dataloader, vng_model, calculate_psnr, f'{name}_vng', example_save_dir):.2f}")
    print(f"  ML: {eval_model(val_dataloader, model, calculate_psnr, f'{name}_ml', example_save_dir):.2f}")


if __name__ == '__main__':
    eval('Kodak')
    eval('McM')
