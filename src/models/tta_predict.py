# src/models/tta_predict.py
import argparse
from pathlib import Path
from src.models.classifier import WasteClassifier
import csv

def main(model_checkpoint, folder, out_csv="tta_results.csv"):
    clf = WasteClassifier(checkpoint=model_checkpoint, tta=True)
    p = Path(folder)
    imgs = sorted(list(p.glob("*.jpg")) + list(p.glob("*.png")))
    rows = []
    for im in imgs:
        label, prob, probs = clf.predict(str(im))
        rows.append([str(im), label, prob] + [float(p) for p in probs])
    # save
    header = ["image", "pred_label", "pred_prob"] + [f"class_{c}" for c in clf.classes]
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print("Saved TTA results to", out_csv)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/resnet18_best.pth")
    ap.add_argument("--folder", default="data/trashnet_split/test")
    ap.add_argument("--out", default="tta_results.csv")
    args = ap.parse_args()
    main(args.model, args.folder, args.out)
