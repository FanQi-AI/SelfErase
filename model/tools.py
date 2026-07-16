import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Union, Optional
import glob

# Auto save figure with auto-incremented filename
def save_image_with_auto_rename(fig, base_path="outputs/out.png", mean_sim=None):
    name, ext = os.path.splitext(base_path)
    folder = os.path.dirname(base_path) or "."
    if mean_sim is not None:
        mean_val = mean_sim.item()  # Get float value
        i = 0
        path = f"{name}_{i}_{mean_val:.4f}{ext}"
        existing_files = os.listdir(folder)
        while any(f.startswith(f"{os.path.basename(name)}_{i}") for f in existing_files):
            i += 1            
            path = f"{name}_{i}_{mean_val:.4f}{ext}"
    else:
        path = base_path
        i = 1
        while os.path.exists(path):
            path = f"{name}_{i}{ext}"
            i += 1
    fig.savefig(path, transparent=True, dpi=150, bbox_inches='tight', pad_inches=0)

def save_top_k(mask, mean_sim=None,H=64, W=64, base_path="/outputs/topk/topk_patch_mask.png"):
    """
    mask: [B, N] Boolean tensor, True means the patch is selected.
    H, W: patch height and width (image grid size)
    """
    B, N = mask.shape

    for b in range(B):
        if B == 2 and b == 0:
            continue
        mask_b = mask[b].reshape(H, W)
        if isinstance(mask_b, torch.Tensor):
            mask_b = mask_b.cpu().numpy()

        # Construct RGBA visualization
        mask_img = np.zeros((H, W, 4), dtype=np.float32)
        mask_img[..., 0] = mask_b            # Red channel
        mask_img[..., 3] = mask_b * 0.4     # Alpha channel (0.8 adjustable)

        # Plotting
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(mask_img)
        ax.axis('off')
        ax.set_facecolor("none")  # Transparent background

        # Save figure
        if mean_sim is not None:
            save_image_with_auto_rename(fig, base_path=base_path,mean_sim=mean_sim)
        else:
            save_image_with_auto_rename(fig, base_path=base_path)
        plt.close(fig)

import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_sim_distribution(sim: torch.Tensor, bins: int = 50, mode: str = "hist", 
                          save_path="/outputs/sim/distribution.png"):
    """
    Plot similarity score distribution and highlight the most frequent bin.
    Styling: paper style, no axis frame.
    """
    sim_np = sim.detach().cpu().numpy().ravel()
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.style.use("seaborn-v0_8-whitegrid")

    base_color = "#80CBC4"   # soft green
    highlight_color = "#F57C00"  # orange for highest frequency bin

    if mode == "hist":
        counts, bins_edges, patches = ax.hist(
            sim_np, bins=bins, color=base_color,
            alpha=0.9, edgecolor="none"
        )
        max_bin = np.argmax(counts)
        patches[max_bin].set_facecolor(highlight_color)
    elif mode == "bar":
        counts, bin_edges = np.histogram(sim_np, bins=bins)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.bar(
            centers, counts, width=(bin_edges[1] - bin_edges[0]),
            color=base_color, alpha=0.9, edgecolor="none"
        )
        max_bin = np.argmax(counts)
        ax.bar(
            centers[max_bin], counts[max_bin],
            width=(bin_edges[1] - bin_edges[0]),
            color=highlight_color, alpha=1.0
        )
    else:
        raise ValueError("mode must be 'hist' or 'bar'")

    # Formatting
    ax.set_xlabel("Similarity score", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_xlim(-0.1, 0.1)
    ax.tick_params(labelsize=10)

    # Remove axis frames for paper-style appearance
    for spine in ax.spines.values():
        spine.set_visible(False)

    save_image_with_auto_rename(fig, base_path=save_path)
    plt.close(fig)


def append_tensor_values(tensor: torch.Tensor, filepath: str, fmt: Optional[str]=None):
    """
    Append tensor values into a text file separated by space.
    - Creates the file if not exists.
    - If file has existing content, space is inserted before new values.
    - Tensor is flattened before writing.
    - Supports GPU tensors (automatically moved to CPU).
    
    Args:
        tensor: torch.Tensor of any shape
        filepath: target file path (e.g., "out.txt")
        fmt: optional format string such as "{:.6f}" for formatting floats.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be torch.Tensor")

    arr = tensor.detach().cpu().numpy().ravel()

    if fmt is None:
        str_vals = [str(x) for x in arr]
    else:
        str_vals = [fmt.format(x) for x in arr]

    need_leading_space = os.path.exists(filepath) and os.path.getsize(filepath) > 0

    with open(filepath, "a", encoding="utf-8") as f:
        if need_leading_space:
            f.write(" ")
        f.write(" ".join(str_vals))


# ====== Test ======
if __name__ == "__main__":
    # Example 1: random vector
    t = torch.randn(10, dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")
    append_tensor_values(t, "numbers.txt", fmt="{:.6f}")   # save with 6 decimal digits

    # Example 2: scalar
    s = torch.tensor(3.14159)
    append_tensor_values(s, "numbers.txt", fmt="{:.6f}")

    # Example 3: append again (automatically space-separated)
    u = torch.tensor([1,2,3], dtype=torch.int64)
    append_tensor_values(u, "numbers.txt")  # integers default to str()
