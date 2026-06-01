import argparse
import copy
import importlib.util
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
    from tqdm import tqdm
    from model import bce_dice_loss, get_device, prepare_episode, validate

    if loss_fn is None:
        loss_fn = bce_dice_loss

    device = device or get_device()
    model = model.to(device)
    history = {"support_loss": [], "query_loss": [], "val_dice": []}

    for step, batch in enumerate(tqdm(train_loader, desc="Reptile"), start=1):
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
            val_metrics = validate(model, current_val_loader, device)
            history["val_dice"].append((step, val_metrics["dice"]))
            tqdm.write(
                f"Step {step:04d} | support loss: {history['support_loss'][-1]:.4f} | "
                f"query loss: {history['query_loss'][-1]:.4f} | val dice: {val_metrics['dice']:.4f}"
            )

    return model, history


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
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", default="reptile_3d_unet.pth")
    parser.add_argument("--history-path", default="reptile_history.json")
    parser.add_argument("--metrics-path", default="reptile_metrics.json")
    return parser.parse_args()


def main():
    args = parse_args()

    import torch
    from model import UNet3D, get_device, test, validate

    dataloader_path = Path(__file__).with_name("dataloader-ben2.py")
    spec = importlib.util.spec_from_file_location("dataloader_ben2", dataloader_path)
    dataloader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dataloader)
    build_3d_dataloader = dataloader.build_3d_dataloader
    build_episode_loader = dataloader.build_episode_loader

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
        return build_3d_dataloader(
            data_dir=args.data_dir,
            split_file=args.val_split,
            batch_size=args.batch_size,
            shuffle=False,
        )

    def make_test_loader():
        return build_3d_dataloader(
            data_dir=args.data_dir,
            split_file=args.test_split,
            batch_size=args.batch_size,
            shuffle=False,
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

    val_metrics = validate(model, make_val_loader(), device=device)
    test_metrics = test(model, make_test_loader(), device=device)

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
