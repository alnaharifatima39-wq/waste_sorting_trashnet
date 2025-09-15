# src/data/check_dataset.py
from pathlib import Path
p = Path("data/trashnet")
if not p.exists():
    print("data/trashnet not found. Put your unzipped TrashNet dataset into data/trashnet/")
else:
    for item in sorted(p.iterdir()):
        if item.is_dir():
            imgs = [f for f in item.iterdir() if f.suffix.lower() in (".jpg",".jpeg",".png")]
            print(f"{item.name}: {len(imgs)} images")
        else:
            print(item.name, "(file)")
