import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import pipeline
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Show SAM2 masks for an image")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument(
        "--model",
        default="facebook/sam2.1-hiera-large",
        help="SAM2 model id for mask generation",
    )
    parser.add_argument(
        "--points-per-batch",
        type=int,
        default=64,
        help="Higher is faster, but uses more VRAM",
    )
    parser.add_argument(
        "--output",
        default="masks_overlay.png",
        help="Path to save colored mask overlay",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="How many masks to print stats for",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = 0 if torch.cuda.is_available() else -1

    generator = pipeline(
        "mask-generation",
        model=args.model,
        device=device,
    )

    image = Image.open(args.image).convert("RGB")
    image_np = np.array(image)
    h, w = image_np.shape[:2]

    outputs = generator(image, points_per_batch=args.points_per_batch, pred_iou_thresh=0.9, stability_score_thresh=0.95, crops_nms_thresh=0.6)

    masks = outputs["masks"]
    print(f"Generated {len(masks)} masks")

    mask_stats = []
    for i, mask in enumerate(masks):
        m = np.array(mask).astype(bool)
        area = int(m.sum())
        frac = area / (h * w)
        ys, xs = np.where(m)
        if area > 0:
            y1, y2 = int(ys.min()), int(ys.max())
            x1, x2 = int(xs.min()), int(xs.max())
            bbox = [x1, y1, x2, y2]
        else:
            bbox = None
        mask_stats.append((i, area, frac, bbox, m))

    # sort by area descending so the biggest masks print first
    mask_stats.sort(key=lambda x: x[1], reverse=True)

    print("\nTop masks by area:")
    for i, area, frac, bbox, _ in mask_stats[: args.topk]:
        print(
            f"mask {i:03d} | area={area:7d} px | "
            f"{frac*100:6.2f}% of image | bbox={bbox}"
        )

    # colored overlay
    rng = np.random.default_rng(0)
    overlay = image_np.astype(np.float32).copy()

    for _, _, _, _, m in mask_stats:
        color = rng.integers(0, 256, size=3)
        overlay[m] = 0.65 * overlay[m] + 0.35 * color

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(args.output)
    print(f"\nSaved overlay to: {args.output}")

    # also save a matplotlib preview with mask indices at mask centers
    plt.figure(figsize=(10, 8))
    plt.imshow(overlay)
    plt.axis("off")
    plt.title(f"{len(masks)} SAM2 masks")

    for i, area, frac, bbox, m in mask_stats[: args.topk]:
        ys, xs = np.where(m)
        if len(xs) == 0:
            continue
        cx = float(xs.mean())
        cy = float(ys.mean())
        plt.text(cx, cy, str(i), fontsize=8)

    preview_path = "masks_preview.png"
    plt.tight_layout()
    plt.savefig(preview_path, bbox_inches="tight", pad_inches=0)
    print(f"Saved preview to: {preview_path}")


if __name__ == "__main__":
    main()