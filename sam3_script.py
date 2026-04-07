import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from transformers import Sam3Model, Sam3Processor


def parse_args():
    parser = argparse.ArgumentParser(description="Show SAM3 masks for an image")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument(
        "--model",
        default="facebook/sam3",
        help="SAM3 model id, e.g. facebook/sam3 or facebook/sam3",
    )
    parser.add_argument(
        "--text-prompt",
        required=True,
        help='Text concept to segment, e.g. "car", "tree", "person"',
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for post-processing",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Mask binarization threshold for post-processing",
    )
    parser.add_argument(
        "--output",
        default="masks_overlay.png",
        help="Path to save colored mask overlay",
    )
    parser.add_argument(
        "--preview",
        default="masks_preview.png",
        help="Path to save matplotlib preview",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="How many masks to print stats for",
    )
    return parser.parse_args()


def to_numpy_mask(mask):
    if isinstance(mask, torch.Tensor):
        return mask.detach().cpu().numpy().astype(bool)
    return np.array(mask).astype(bool)


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading model: {args.model}")
    processor = Sam3Processor.from_pretrained(args.model)
    model = Sam3Model.from_pretrained(args.model).to(device)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    image_np = np.array(image)
    h, w = image_np.shape[:2]

    inputs = processor(
        images=image,
        text=args.text_prompt,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=args.threshold,
        mask_threshold=args.mask_threshold,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]

    masks = results.get("masks", [])
    boxes = results.get("boxes", None)
    scores = results.get("scores", None)

    num_masks = len(masks)
    print(f"Generated {num_masks} masks for prompt: {args.text_prompt!r}")

    mask_stats = []
    for i, mask in enumerate(masks):
        m = to_numpy_mask(mask)
        area = int(m.sum())
        frac = area / (h * w) if h * w > 0 else 0.0

        ys, xs = np.where(m)
        if area > 0:
            y1, y2 = int(ys.min()), int(ys.max())
            x1, x2 = int(xs.min()), int(xs.max())
            bbox_from_mask = [x1, y1, x2, y2]
        else:
            bbox_from_mask = None

        box = None
        if boxes is not None and i < len(boxes):
            b = boxes[i]
            if isinstance(b, torch.Tensor):
                b = b.detach().cpu().tolist()
            box = [int(round(v)) for v in b]

        score = None
        if scores is not None and i < len(scores):
            s = scores[i]
            if isinstance(s, torch.Tensor):
                s = float(s.detach().cpu().item())
            else:
                s = float(s)

        mask_stats.append((i, area, frac, box, bbox_from_mask, score, m))

    mask_stats.sort(key=lambda x: x[1], reverse=True)

    print("\nTop masks by area:")
    for i, area, frac, box, bbox_from_mask, score, _ in mask_stats[: args.topk]:
        score_str = f"{score:.4f}" if score is not None else "N/A"
        print(
            f"mask {i:03d} | "
            f"score={score_str} | "
            f"area={area:7d} px | "
            f"{frac*100:6.2f}% of image | "
            f"box={box} | "
            f"mask_bbox={bbox_from_mask}"
        )

    rng = np.random.default_rng(0)
    overlay = image_np.astype(np.float32).copy()

    for _, _, _, _, _, _, m in mask_stats:
        color = rng.integers(0, 256, size=3)
        overlay[m] = 0.65 * overlay[m] + 0.35 * color

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(args.output)
    print(f"\nSaved overlay to: {args.output}")

    plt.figure(figsize=(10, 8))
    plt.imshow(overlay)
    plt.axis("off")
    plt.title(f"{num_masks} SAM3 masks for: {args.text_prompt}")

    for i, area, frac, box, bbox_from_mask, score, m in mask_stats[: args.topk]:
        ys, xs = np.where(m)
        if len(xs) == 0:
            continue
        cx = float(xs.mean())
        cy = float(ys.mean())
        label = str(i)
        if score is not None:
            label += f" ({score:.2f})"
        plt.text(cx, cy, label, fontsize=8)

    plt.tight_layout()
    plt.savefig(args.preview, bbox_inches="tight", pad_inches=0)
    print(f"Saved preview to: {args.preview}")


if __name__ == "__main__":
    main()