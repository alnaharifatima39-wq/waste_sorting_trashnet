from PIL import Image
import cv2
import argparse
import time
from src.models.classifier import WasteClassifier

def main(model_path, camera_id):
    clf = WasteClassifier(model_path)
    if not clf.is_ready():
        print("Model not loaded! Exiting.")
        return

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Could not open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite("temp.jpg", frame)

        # ✅ Open image with PIL, not just pass filename
        pil_img = Image.open("temp.jpg").convert("RGB")
        preds = clf.predict(pil_img, topk=1)
        label, prob = preds[0]

        cv2.putText(frame, f"{label}: {prob:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Waste Sorting Demo", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/resnet_trashnet.pth")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    main(args.model_path, args.camera)
