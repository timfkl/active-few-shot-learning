import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from config import (
    IN_CHANNELS,
    OUT_CHANNELS,
    POS_WEIGHT,
    UNET_FEATURES,
)


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups != 0:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, features=UNET_FEATURES):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        channels = in_channels
        for feature in features:
            self.encoders.append(ConvBlock3D(channels, feature))
            self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            channels = feature

        self.bottleneck = ConvBlock3D(features[-1], features[-1] * 2)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        channels = features[-1] * 2
        for feature in reversed(features):
            self.upconvs.append(nn.ConvTranspose3d(channels, feature, kernel_size=2, stride=2))
            self.decoders.append(ConvBlock3D(feature * 2, feature))
            channels = feature

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = decoder(x)

        return self.final_conv(x)


def prepare_batch(batch, device, binary_mask=True):
    if isinstance(batch, dict):
        images = batch["image"]
        masks = batch["mask"]
    else:
        images, masks = batch[:2]

    images = images.float()
    masks = masks.float()

    if images.ndim == 4:
        images = images.unsqueeze(1)
    if masks.ndim == 4:
        masks = masks.unsqueeze(1)

    # dataloader-ben2 returns [B, 1, X, Y, D]; Conv3d expects [B, 1, D, X, Y].
    if images.ndim == 5 and images.shape[-1] < images.shape[2]:
        images = images.permute(0, 1, 4, 2, 3).contiguous()
        masks = masks.permute(0, 1, 4, 2, 3).contiguous()

    if binary_mask:
        masks = (masks > 0).float()

    return images.to(device), masks.to(device)


def prepare_episode(batch, device):
    if isinstance(batch, dict) and "support" in batch and "query" in batch:
        support_images, support_masks = prepare_batch(batch["support"], device)
        query_images, query_masks = prepare_batch(batch["query"], device)
        return support_images, support_masks, query_images, query_masks

    images, masks = prepare_batch(batch, device)
    return split_support_query(images, masks)


def dice_score(pred_logits, target, threshold=0.5, eps=1e-6):
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    dims = tuple(range(1, pred.ndim))
    intersection = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims)
    return ((2.0 * intersection + eps) / (union + eps)).mean()


def dice_loss(pred_logits, target, eps=1e-6):
    pred = torch.sigmoid(pred_logits)
    dims = tuple(range(1, pred.ndim))
    intersection = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims)
    return 1.0 - ((2.0 * intersection + eps) / (union + eps)).mean()


def bce_dice_loss(pred_logits, target, pos_weight=POS_WEIGHT):
    weight = torch.tensor([pos_weight], device=pred_logits.device)
    bce = F.binary_cross_entropy_with_logits(pred_logits, target, pos_weight=weight)
    return 0.5 * bce + 0.5 * dice_loss(pred_logits, target)


@torch.no_grad()
def validate(model, val_loader, device=None, loss_fn=bce_dice_loss):
    device = device or get_device()
    model.eval()
    losses = []
    dices = []

    for batch in tqdm(val_loader, desc="Validate", leave=False):
        images, masks = prepare_batch(batch, device)
        preds = model(images)
        loss = loss_fn(preds, masks)
        dice = dice_score(preds, masks)

        if torch.isfinite(loss):
            losses.append(loss.item())
        if torch.isfinite(dice):
            dices.append(dice.item())

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "dice": float(np.mean(dices)) if dices else float("nan"),
    }


@torch.no_grad()
def test(model, test_loader, device=None):
    metrics = validate(model, test_loader, device=device)
    print(f"Test loss: {metrics['loss']:.4f} | Test Dice: {metrics['dice']:.4f}")
    return metrics


def split_support_query(images, masks):
    if images.size(0) < 2:
        return images, masks, images, masks

    split = images.size(0) // 2
    return images[:split], masks[:split], images[split:], masks[split:]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
