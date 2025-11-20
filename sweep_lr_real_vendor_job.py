import argparse
import os
import time
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
# Import autocast and GradScaler
# For PyTorch 2.1+, autocast is in torch.amp, but GradScaler is still in torch.cuda.amp
# For older versions, both are in torch.cuda.amp
# try:
#     from torch.amp import autocast as autocast_new
#     USE_NEW_AUTOCAST = True
# except ImportError:
#     from torch.cuda.amp import autocast as autocast_old
#     USE_NEW_AUTOCAST = False

# from torch.cuda.amp import GradScaler
# from torch.amp import autocast, GradScaler 

# try:
#     # Newer PyTorch versions
#     from torch.cuda.amp import autocast, GradScaler
# except ImportError:
#     # Fallback for ROCm or older builds
#     from torch.amp import autocast
#     from torch.cuda.amp import GradScaler

from real_tiny_gpt_model import TinyGPTReal


# ---------- tiny text dataset (same idea as before) ----------

def get_tiny_text() -> str:
    # You can replace this with your Shakespeare snippet / any corpus.
    return (
        "To be, or not to be, that is the question:\n"
        "Whether 'tis nobler in the mind to suffer\n"
        "The slings and arrows of outrageous fortune,\n"
        "Or to take arms against a sea of troubles\n"
        "And by opposing end them.\n"
    )


class CharDataset(torch.utils.data.Dataset):
    def __init__(self, text: str, block_size: int):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)
        self.block_size = block_size
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


# ---------- training ----------

def pick_precision(precision: str) -> Tuple[torch.dtype, bool]:
    precision = precision.lower()
    if precision == "fp16":
        return torch.float16, True
    if precision == "bf16":
        return torch.bfloat16, True
    if precision == "fp32":
        return torch.float32, False
    raise ValueError(f"Unknown precision {precision}")


def train_one_config(
    vendor: str,
    precision: str,
    lr: float,
    sub_scale: float,
    seed: int,
    max_steps: int,
    batch_size: int,
    block_size: int,
    log_interval: int,
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)

    print(" =====================================")
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU device_name ==>", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
    print(" =====================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("torch device = " + str(device) )
    print("device.type = " + str(device.type) )

    torch.manual_seed(seed)
    if device.type == "cuda":
        print("============= hello CUDA ============")
        torch.cuda.manual_seed_all(seed)

    amp_dtype, use_amp = pick_precision(precision)

    # Dataset / loader
    text = get_tiny_text()
    dataset = CharDataset(text, block_size)
    vocab_size = dataset.vocab_size
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    data_iter = iter(loader)

    # Model
    model = TinyGPTReal(
        vocab_size=vocab_size,
        d_model=256,
        n_layer=4,
        n_heads=4,
        d_ff=1024,
        block_size=block_size,
        sub_scale=sub_scale,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    if use_amp:
        # GradScaler API is consistent across versions
        # scaler = GradScaler(enabled=True)
        scaler = torch.amp.GradScaler(device="cuda")
    else:
        scaler = None

    device_name = (
        torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    )
    if device.type == "cuda":
        device_name = torch.cuda.get_device_properties(0).name
    else:
        device_name = "cpu"
    # print(" ========> device_name =", device_name)
    losses = []
    diverged = False
    best_loss = float("inf")
    t0 = time.time()

    for step in range(1, max_steps + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            # NVIDIA or ROCm GPU
            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        else:
            # CPU fallback
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))


            if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 100.0:
                diverged = True
                losses.append(float("nan"))
                print(f"[{vendor}] step {step}: diverged (loss={loss.item():.3e})")
                break

        if use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)
        if loss_val < best_loss:
            best_loss = loss_val

        if step % log_interval == 0 or step == 1:
            elapsed = time.time() - t0
            print(
                f"[{vendor}] step {step:5d} | loss {loss_val:.4f} | best {best_loss:.4f} | "
                f"lr {lr:g} | sub_scale {sub_scale:g} | elapsed {elapsed:.1f}s"
            )

    result = dict(
        vendor=vendor,
        precision=precision,
        device_name=device_name,
        lr=lr,
        sub_scale=sub_scale,
        seed=seed,
        max_steps=max_steps,
        batch_size=batch_size,
        diverged=diverged,
        final_loss=float(losses[-1]) if losses and not diverged else float("inf"),
        best_loss=float(best_loss),
        losses=losses,
    )

    fname = (
        f"result_vendor={vendor}_prec={precision}_lr={lr:g}_sub={sub_scale:g}_seed={seed}.pkl"
    )
    out_path = os.path.join(output_dir, fname)
    torch.save(result, out_path)
    print("Saved", out_path)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vendor", type=str, required=True,
                   help="Logical vendor tag (e.g., fp16_a100, fp16_mi250x)")
    p.add_argument("--precision", type=str, default="fp16",
                   choices=["fp16", "bf16", "fp32"])
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--sub-scale", type=float, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--output-dir", type=str, required=True)
    args = p.parse_args()

    try:
        train_one_config(
            vendor=args.vendor,
            precision=args.precision,
            lr=args.lr,
            sub_scale=args.sub_scale,
            seed=args.seed,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            block_size=args.block_size,
            log_interval=args.log_interval,
            output_dir=args.output_dir,
        )
    except Exception as e:
        print("ERROR:", e)
        # Save a minimal error record so merge scripts don't break.
        os.makedirs(args.output_dir, exist_ok=True)
        result = dict(
            vendor=args.vendor,
            precision=args.precision,
            lr=args.lr,
            sub_scale=args.sub_scale,
            seed=args.seed,
            error=str(e),
            diverged=True,
            final_loss=float("inf"),
            best_loss=float("inf"),
            losses=[],
        )
        fname = (
            f"result_vendor={args.vendor}_prec={args.precision}_lr={args.lr:g}"
            f"_sub={args.sub_scale:g}_seed={args.seed}_ERROR.pkl"
        )
        out_path = os.path.join(args.output_dir, fname)
        torch.save(result, out_path)
        print("Saved error result to", out_path)


if __name__ == "__main__":
    main()

