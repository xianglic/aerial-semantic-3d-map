#!/usr/bin/env python3
import shutil
from pathlib import Path
import argparse

def select_images_every_n(source_dir: Path, target_dir: Path, step: int):
    target_dir.mkdir(parents=True, exist_ok=True)

    jpg_files = sorted(source_dir.glob("*.jpg"))

    if not jpg_files:
        print(f"No .jpg files found in: {source_dir}")
        return

    selected_count = 0
    for i in range(0, len(jpg_files), step):
        source_file = jpg_files[i]
        target_file = target_dir / source_file.name
        shutil.copy2(source_file, target_file)
        selected_count += 1
        print(f"Selected: {source_file.name}")

    print(f"\nSelected {selected_count} images")
    print(f"From {len(jpg_files)} images, picked every {step}")
    print(f"Saved to: {target_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Copy every Nth JPG from a source folder to a target folder."
    )
    parser.add_argument("source_dir", type=Path, help="Folder containing JPG images")
    parser.add_argument("target_dir", type=Path, help="Folder to save selected images")
    parser.add_argument(
        "--step",
        type=int,
        default=100,
        help="Pick every Nth image (default: 100)",
    )

    args = parser.parse_args()

    if not args.source_dir.exists():
        print(f"Source folder does not exist: {args.source_dir}")
        return

    if not args.source_dir.is_dir():
        print(f"Source path is not a folder: {args.source_dir}")
        return

    if args.step <= 0:
        print("--step must be a positive integer")
        return

    select_images_every_n(args.source_dir, args.target_dir, args.step)

if __name__ == "__main__":
    main()
