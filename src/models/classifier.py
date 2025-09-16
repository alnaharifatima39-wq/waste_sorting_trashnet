# src/models/classifier.py
import torch
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import torch.nn.functional as F

class WasteClassifier:
    def __init__(self, model_path="models/resnet_trashnet.pth", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path)
        self.model = None
        self.classes = None
        self._load_model()

        self.tf = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    def _load_model(self):
        if not self.model_path.exists():
            # model missing — leave model as None
            return
        ckpt = torch.load(self.model_path, map_location=self.device)
        self.classes = ckpt.get("classes", None)
        # Build the network skeleton
        backbone = models.resnet18(weights=None)
        backbone.fc = torch.nn.Linear(backbone.fc.in_features, len(self.classes))
        backbone.load_state_dict(ckpt["model_state"])
        backbone.to(self.device)
        backbone.eval()
        self.model = backbone

    def is_ready(self):
        return self.model is not None and self.classes is not None

    def predict(self, pil_image, topk=5):
        """
        Input: PIL Image
        Returns: list of tuples [(class, prob), ...] sorted desc
        """
        if not self.is_ready():
            raise RuntimeError("Model not loaded. Place trained model at: " + str(self.model_path))

        x = self.tf(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        # pair classes with probs
        pairs = list(zip(self.classes, probs))
        pairs_sorted = sorted(pairs, key=lambda t: t[1], reverse=True)
        return pairs_sorted[:topk]
