import argparse
import copy
import json
import random
from pathlib import Path

import config


def reptile_train(
    model,
    train_loader,
    val_loader=None,
    outer_steps=config.REPTILE_OUTER_STEPS,
    inner_steps=config.REPTILE_INNER_STEPS,
    inner_lr=config.REPTILE_INNER_LR,
    outer_lr=config.REPTILE_OUTER_LR,
    device=None,
    loss_fn=None,
    val_interval=config.REPTILE_VAL_INTERVAL,
):
    import numpy as np
    import torch
    import torch.optim as optim
    from tqdm.auto import tqdm
    from model import bce_dice_loss, get_device, prepare_episode

    if loss_fn is None:
        loss_fn = bce_dice_loss

    device = device or get_device()
    model = model.to(device)
    history = {"support_loss": [], "query_loss": [], "val_dice": []}

    progress_bar = tqdm(train_loader, total=outer_steps, desc="Reptile", dynamic_ncols=True)
    for step, batch in enumerate(progress_bar, start=1):
        if step > outer_steps:
            break

        support_images, support_masks, query_images, query_masks = prepare_episode(batch, device)

        task_model = copy.deepcopy(model).to(device)
        task_model.train()
        inner_optimizer = optim.SGD(task_model.parameters(), lr=inner_lr)

        support_losses = []
        for _ in range(inner_steps):
            inner_optimizer.zero_grad()
            preds = task_model(support_images)
            loss = loss_fn(preds, support_masks)

            if not torch.isfinite(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(task_model.parameters(), 1.0)
            inner_optimizer.step()
            support_losses.append(loss.item())

        with torch.no_grad():
            task_model.eval()
            query_loss = loss_fn(task_model(query_images), query_masks)
            progress = (step - 1) / max(outer_steps - 1, 1)
            step_lr = outer_lr * (1.0 - progress)

            for param, task_param in zip(model.parameters(), task_model.parameters()):
                param.add_(step_lr * (task_param.data - param.data))

        history["support_loss"].append(float(np.mean(support_losses)) if support_losses else float("nan"))
        history["query_loss"].append(float(query_loss.item()) if torch.isfinite(query_loss) else float("nan"))

        if val_loader is not None and step % val_interval == 0:
            current_val_loader = val_loader() if callable(val_loader) else val_loader
            val_metrics = evaluate_reptile(
                model,
                current_val_loader,
                inner_steps=inner_steps,
                inner_lr=inner_lr,
                device=device,
                loss_fn=loss_fn,
                show_progress=False,
            )
            history["val_dice"].append((step, val_metrics["dice"]))
            progress_bar.set_postfix(
                support_loss=f"{history['support_loss'][-1]:.4f}",
                query_loss=f"{history['query_loss'][-1]:.4f}",
                val_dice=f"{val_metrics['dice']:.4f}",
            )
        else:
            progress_bar.set_postfix(
                support_loss=f"{history['support_loss'][-1]:.4f}",
                query_loss=f"{history['query_loss'][-1]:.4f}",
            )

    return model, history


def evaluate_reptile(
    model,
    episode_loader,
    inner_steps=config.REPTILE_INNER_STEPS,
    inner_lr=config.REPTILE_INNER_LR,
    device=None,
    loss_fn=None,
    show_progress=True,
):
    import numpy as np
    import torch
    import torch.optim as optim
    from tqdm.auto import tqdm
    from model import bce_dice_loss, dice_score, get_device, prepare_episode

    if loss_fn is None:
        loss_fn = bce_dice_loss

    device = device or get_device()
    model = model.to(device)
    losses = []
    dices = []

    for episode in tqdm(episode_loader, desc="Evaluate Reptile", leave=False, disable=not show_progress):
        support_images, support_masks, query_images, query_masks = prepare_episode(episode, device)

        adapted_model = copy.deepcopy(model).to(device)
        adapted_model.train()
        optimizer = optim.SGD(adapted_model.parameters(), lr=inner_lr)

        for _ in range(inner_steps):
            optimizer.zero_grad()
            support_loss = loss_fn(adapted_model(support_images), support_masks)
            if not torch.isfinite(support_loss):
                continue
            support_loss.backward()
            torch.nn.utils.clip_grad_norm_(adapted_model.parameters(), 1.0)
            optimizer.step()

        adapted_model.eval()
        with torch.no_grad():
            query_preds = adapted_model(query_images)
            query_loss = loss_fn(query_preds, query_masks)
            query_dice = dice_score(query_preds, query_masks)

        if torch.isfinite(query_loss):
            losses.append(query_loss.item())
        if torch.isfinite(query_dice):
            dices.append(query_dice.item())

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "dice": float(np.mean(dices)) if dices else float("nan"),
    }


def set_seed(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a 3D U-Net with Reptile.")
    parser.add_argument("--data-dir", default=config.RESIZED_DATA_DIR)
    parser.add_argument("--train-split", default="train.txt")
    parser.add_argument("--val-split", default="val.txt")
    parser.add_argument("--test-split", default="test.txt")
    parser.add_argument("--n-support", type=int, default=1)
    parser.add_argument("--n-query", type=int, default=1)
    parser.add_argument("--outer-steps", type=int, default=config.REPTILE_OUTER_STEPS)
    parser.add_argument("--inner-steps", type=int, default=config.REPTILE_INNER_STEPS)
    parser.add_argument("--inner-lr", type=float, default=config.REPTILE_INNER_LR)
    parser.add_argument("--outer-lr", type=float, default=config.REPTILE_OUTER_LR)
    parser.add_argument("--val-interval", type=int, default=config.REPTILE_VAL_INTERVAL)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", default="reptile_3d_unet.pth")
    parser.add_argument("--history-path", default="reptile_history.json")
    parser.add_argument("--metrics-path", default="reptile_metrics.json")
    return parser.parse_args()


def main():
    args = parse_args()

    import torch
    from dataloader_ben2 import build_episode_loader
    from model import UNet3D, get_device

    set_seed(args.seed)

    device = get_device()
    print(f"Device: {device}")

    train_loader = build_episode_loader(
        data_dir=args.data_dir,
        split_file=args.train_split,
        n_support=args.n_support,
        n_query=args.n_query,
        episodes=args.outer_steps,
    )
    def make_val_loader():
        return build_episode_loader(
            data_dir=args.data_dir,
            split_file=args.val_split,
            n_support=args.n_support,
            n_query=args.n_query,
            episodes=args.eval_episodes,
        )

    def make_test_loader():
        return build_episode_loader(
            data_dir=args.data_dir,
            split_file=args.test_split,
            n_support=args.n_support,
            n_query=args.n_query,
            episodes=args.eval_episodes,
        )

    model = UNet3D().to(device)
    model, history = reptile_train(
        model,
        train_loader,
        val_loader=make_val_loader,
        outer_steps=args.outer_steps,
        inner_steps=args.inner_steps,
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr,
        val_interval=args.val_interval,
        device=device,
    )

    torch.save(model.state_dict(), args.model_path)

    val_metrics = evaluate_reptile(
        model,
        make_val_loader(),
        inner_steps=args.inner_steps,
        inner_lr=args.inner_lr,
        device=device,
    )
    test_metrics = evaluate_reptile(
        model,
        make_test_loader(),
        inner_steps=args.inner_steps,
        inner_lr=args.inner_lr,
        device=device,
    )
    print(f"Validation loss: {val_metrics['loss']:.4f} | Validation Dice: {val_metrics['dice']:.4f}")
    print(f"Test loss: {test_metrics['loss']:.4f} | Test Dice: {test_metrics['dice']:.4f}")

    Path(args.history_path).write_text(json.dumps(history, indent=2), encoding="utf-8")
    Path(args.metrics_path).write_text(
        json.dumps({"validation": val_metrics, "test": test_metrics}, indent=2),
        encoding="utf-8",
    )

    print(f"Saved model to {args.model_path}")
    print(f"Saved history to {args.history_path}")
    print(f"Saved metrics to {args.metrics_path}")


if __name__ == "__main__":
    main()
