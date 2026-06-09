import copy

import torch
import torch.nn as nn
import torch.optim as optim

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── 3D UNet model ───────────────────────────────────────────────────
class ConvBlock(nn.Module):
    """Two 3D convolution layers used throughout the U-Net."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    """Small 3D U-Net for volumetric image segmentation."""

    def __init__(self, in_channels=1, out_channels=1, features=(8, 16, 32)):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        channels = in_channels
        for feature in features:
            self.encoders.append(ConvBlock(channels, feature))
            self.pools.append(nn.MaxPool3d(2))
            channels = feature

        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        channels = features[-1] * 2
        for feature in reversed(features):
            self.upconvs.append(nn.ConvTranspose3d(channels, feature, kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(feature * 2, feature))
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
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
        return self.final_conv(x)


# ── Loss functions and metrics ──────────────────────────────────────
def dice_score(pred_logits, target, threshold=0.5, eps=1e-6):
    """Calculate Dice score after thresholding predicted probabilities."""
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    dims = tuple(range(1, pred.ndim))
    intersection = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims)
    return ((2.0 * intersection + eps) / (union + eps)).mean()


def dice_loss(pred_logits, target, eps=1e-6):
    """Soft Dice loss for segmentation."""
    pred = torch.sigmoid(pred_logits)
    dims = tuple(range(1, pred.ndim))
    intersection = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims)
    return 1.0 - ((2.0 * intersection + eps) / (union + eps)).mean()


def bce_dice_loss(pred_logits, target, pos_weight=5.0):
    """Hybrid loss: weighted BCE for imbalance plus Dice loss for overlap."""
    weight = torch.tensor([pos_weight], device=pred_logits.device)
    bce = nn.functional.binary_cross_entropy_with_logits(
        pred_logits,
        target,
        pos_weight=weight,
    )
    return (0.5 * bce) + (0.5 * dice_loss(pred_logits, target))


# ── Helper functions ───────────────────────────────────────────────
def prepare_batch(images, masks, device):
    """Convert numpy batches from the data generator into PyTorch 3D tensors."""
    images = torch.from_numpy(images).float().to(device)
    masks = torch.from_numpy(masks).float().to(device)

    # Generator gives images as (B, C, H, W, D); Conv3d expects (B, C, D, H, W).
    images = images.permute(0, 1, 4, 2, 3)
    if masks.ndim == 4:
        masks = masks[:, None, ...]
    masks = masks.permute(0, 1, 4, 2, 3)
    return images, masks


def split_support_query(images, masks, n_support):
    """Split a batch into support samples and query samples."""
    if len(images) <= n_support:
        raise ValueError("Batch must contain support and query samples.")

    support_images = images[:n_support]
    support_masks = masks[:n_support]
    query_images = images[n_support:]
    query_masks = masks[n_support:]

    return support_images, support_masks, query_images, query_masks


# ── Reptile helper functions ─────────────────────────────────────────
def train_reptile(model, train_loader, val_loader=None, n_support=1, outer_steps=100, inner_steps=5, inner_lr=1e-3, outer_lr=0.1, val_interval=10, device=None):
    """Train a Reptile initialization using batches from external loaders."""
    device = device or get_device()
    model = model.to(device)
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
    }

    for step in range(outer_steps):
        # The loader comes from data_loader.py and yields numpy image/mask batches.
        images, masks = next(train_loader)
        images, masks = prepare_batch(images, masks, device)

        support_images, support_masks, _, _ = split_support_query(images, masks, n_support)

        adapted_model = copy.deepcopy(model).to(device)
        adapted_model.train()
        optimizer = optim.SGD(adapted_model.parameters(), lr=inner_lr)

        loss = None
        for _ in range(inner_steps):
            optimizer.zero_grad()
            loss = bce_dice_loss(adapted_model(support_images), support_masks)
            loss.backward()
            optimizer.step()

        progress = step / max(outer_steps - 1, 1)
        step_outer_lr = outer_lr * (1.0 - progress)

        # Reptile update: move the original model toward the adapted model.
        with torch.no_grad():
            for param, adapted_param in zip(model.parameters(), adapted_model.parameters()):
                param.add_(step_outer_lr * (adapted_param - param))

        history["train_loss"].append(float(loss.item()))

        if val_loader is not None and (step + 1) % val_interval == 0:
            val_images, val_masks = next(val_loader)
            val_support_images, val_support_masks, val_query_images, val_query_masks = (
                split_support_query(val_images, val_masks, n_support)
            )

            val_model = adapt_on_support(model, val_support_images, val_support_masks, inner_steps=inner_steps, inner_lr=inner_lr, device=device)
            val_metrics = evaluate_on_query(val_model, val_query_images, val_query_masks, device=device)
            history["val_loss"].append(val_metrics["loss"])
            history["val_dice"].append(val_metrics["dice"])

    return model, history

def adapt_on_support(model, support_images, support_masks, inner_steps=5, inner_lr=1e-3, device=None):
    """Adapt a copy of the model using selected support images and masks."""
    device = device or get_device()
    support_x, support_y = prepare_batch(support_images, support_masks, device)

    adapted_model = copy.deepcopy(model).to(device)
    adapted_model.train()
    optimizer = optim.SGD(adapted_model.parameters(), lr=inner_lr)

    for _ in range(inner_steps):
        optimizer.zero_grad()
        loss = bce_dice_loss(adapted_model(support_x), support_y)
        loss.backward()
        optimizer.step()

    return adapted_model

@torch.no_grad()
def evaluate_on_query(model, query_images, query_masks, device=None):
    """Evaluate an adapted model on query images and masks."""
    device = device or get_device()
    query_x, query_y = prepare_batch(query_images, query_masks, device)

    model = model.to(device)
    model.eval()
    logits = model(query_x)

    return {
        "loss": float(bce_dice_loss(logits, query_y).item()),
        "dice": float(dice_score(logits, query_y).item()),
    }
