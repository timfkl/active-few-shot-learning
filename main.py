import torch

from resize import resize_dataset
from split import find_case_ids, split_case_ids, write_case_ids
from data_loader import build_nifti_batch_generator
from reptile import UNet3D, train_reptile, get_device
from RL import run_rl_active_selection

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR        = "data"
RESIZE_DIR      = "data-resize"
WEIGHTS_PATH    = "reptile_init.pt"

RESIZE_DIMS     = (256, 256, 32)
TRAIN_RATIO     = 0.7
VAL_RATIO       = 0.15
TEST_RATIO      = 0.15
SEED            = 42
TASK_NUMBER     = 1

REPTILE_BATCH   = 4
N_SUPPORT       = 4
OUTER_STEPS     = 100
INNER_STEPS     = 5
INNER_LR        = 1e-3
OUTER_LR        = 0.1
VAL_INTERVAL    = 10

RL_BATCH        = 16
RL_TRAIN_TRIALS = 10
RL_STEPS        = 256
RL_EVAL_STEPS   = 100


def main():
    # ── 1. Resize ────────────────────────────────────────────────────────────
    print("=== Step 1: Resizing dataset ===")
    resize_dataset(input_dir=DATA_DIR, output_dir=RESIZE_DIR, dimensions=RESIZE_DIMS)

    # ── 2. Split ─────────────────────────────────────────────────────────────
    print("=== Step 2: Splitting dataset ===")
    case_ids = find_case_ids(RESIZE_DIR)
    train_ids, val_ids, test_ids = split_case_ids(
        case_ids, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, seed=SEED
    )
    write_case_ids(train_ids, "train.txt")
    write_case_ids(val_ids,   "val.txt")
    write_case_ids(test_ids,  "test.txt")
    print(f"Split: {len(train_ids)} train | {len(val_ids)} val | {len(test_ids)} test")

    # ── 3. Train Reptile ─────────────────────────────────────────────────────
    print("=== Step 3: Training Reptile ===")
    train_loader = build_nifti_batch_generator(
        batch_size=REPTILE_BATCH,
        images_dir=RESIZE_DIR,
        masks_dir=RESIZE_DIR,
        normalize=True,
        task_number=TASK_NUMBER,
    )
    val_loader = build_nifti_batch_generator(
        batch_size=REPTILE_BATCH,
        images_dir=RESIZE_DIR,
        masks_dir=RESIZE_DIR,
        normalize=True,
        task_number=TASK_NUMBER,
    )

    model, history = train_reptile(
        model=UNet3D(),
        train_loader=train_loader,
        val_loader=val_loader,
        n_support=N_SUPPORT,
        outer_steps=OUTER_STEPS,
        inner_steps=INNER_STEPS,
        inner_lr=INNER_LR,
        outer_lr=OUTER_LR,
        val_interval=VAL_INTERVAL,
        device=get_device(),
    )

    # ── 4. Save weights ───────────────────────────────────────────────────────
    print(f"=== Step 4: Saving weights to {WEIGHTS_PATH} ===")
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"Final train loss : {history['train_loss'][-1]:.4f}")
    if history["val_dice"]:
        print(f"Final val dice   : {history['val_dice'][-1]:.4f}")

    # ── 5. Run RL ─────────────────────────────────────────────────────────────
    # Imported here because RL.py loads the weights file at module level,
    # so it must be imported after the weights are saved above.
    print("=== Step 5: Running RL active selection ===")
    dice_history = run_rl_active_selection(
        batch_size=RL_BATCH,
        n_support=N_SUPPORT,
        task_number=TASK_NUMBER,
        train_trials=RL_TRAIN_TRIALS,
        steps_per_trial=RL_STEPS,
        eval_steps=RL_EVAL_STEPS,
    )
    print("RL dice history:", [f"{d:.4f}" for d in dice_history])


if __name__ == "__main__":
    main()
