from src.models.classifier import WasteClassifier

if __name__=="__main__":
    clf = WasteClassifier("models/resnet_trashnet.pth")
    label, prob = clf.predict("data/trashnet_split/test/metal/metal82.jpg")
    print("Prediction:", label, prob)
