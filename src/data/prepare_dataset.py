# src/data/prepare_dataset.py
import argparse
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def find_classes_by_folder(input_dir: Path):
    # return immediate subfolder names that contain images
    classes = []
    for p in sorted(input_dir.iterdir()):
        if p.is_dir():
            if any((pglob.suffix.lower() in IMG_EXTS) for pglob in p.glob("*")):
                classes.append(p.name)
    return classes

def find_classes_recursive(input_dir: Path):
    # fallback: scan recursively and use parent folder names of images
    classes = set()
    for img in input_dir.rglob("*"):
        if img.suffix.lower() in IMG_EXTS and img.is_file():
            classes.add(img.parent.name)
    return sorted(classes)

def gather_files_for_class(input_dir: Path, cls_name: str):
    # Prefer direct folder input_dir/cls_name/*
    folder = input_dir / cls_name
    if folder.exists() and folder.is_dir():
        files = [f for f in folder.glob("*") if f.suffix.lower() in IMG_EXTS]
        return files
    # Otherwise search recursively for files whose parent folder name is cls_name
    files = [f for f in input_dir.rglob("*") if f.is_file() and f.suffix.lower() in IMG_EXTS and f.parent.name == cls_name]
    return files

def prepare(input_dir="data/trashnet", output_dir="data/trashnet_split", val_size=0.15, test_size=0.15, seed=42):
    inp = Path(input_dir)
    out = Path(output_dir)

    if not inp.exists():
        raise FileNotFoundError(f"Input folder not found: {inp.resolve()}")

    # remove old split if exists
    if out.exists():
        shutil.rmtree(out)
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "val").mkdir(parents=True, exist_ok=True)
    (out / "test").mkdir(parents=True, exist_ok=True)

    classes = find_classes_by_folder(inp)
    if not classes:
        classes = find_classes_recursive(inp)

    if not classes:
        raise RuntimeError(f"No classes/images found under {inp}. Check your dataset layout.")

    print("Detected classes:", classes)

    for cls in classes:
        files = gather_files_for_class(inp, cls)
        if not files:
            print(f"Warning: no images found for class '{cls}', skipping.")
            continue

        train_files, rest = train_test_split(files, test_size=val_size + test_size, random_state=seed)
        val_rel = val_size / (val_size + test_size)
        val_files, test_files = train_test_split(rest, test_size=1 - val_rel, random_state=seed)

        # create class folders before copying
        (out / "train" / cls).mkdir(parents=True, exist_ok=True)
        (out / "val" / cls).mkdir(parents=True, exist_ok=True)
        (out / "test" / cls).mkdir(parents=True, exist_ok=True)

        for f in train_files:
            shutil.copy(f, out / "train" / cls / f.name)
        for f in val_files:
            shutil.copy(f, out / "val" / cls / f.name)
        for f in test_files:
            shutil.copy(f, out / "test" / cls / f.name)

    print("Dataset prepared at:", out.resolve())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/trashnet", help="Path to raw dataset root")
    ap.add_argument("--output", default="data/trashnet_split", help="Path to output split root")
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    prepare(args.input, args.output, args.val_size, args.test_size, args.seed)
