# src/data/check_split.py
from pathlib import Path
p = Path("data/trashnet_split")
if not p.exists():
    print("Split folder not found: data/trashnet_split")
else:
    for split in ("train","val","test"):
        sp = p / split
        if not sp.exists():
            print(f"{split}: MISSING")
            continue
        print(f"=== {split} ===")
        for cls in sorted(sp.iterdir()):
            if cls.is_dir():
                count = len([f for f in cls.glob("*") if f.suffix.lower() in (".jpg",".jpeg",".png")])
                print(f"{cls.name}: {count}")
