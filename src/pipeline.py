"""BookLens AI End-to-End Inference & Curation Pipeline."""
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from .model import build_resnet18_transfer, CLASSES, CLASS_KR
from .curator import BookLensNLPCurator

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class BookLensPipeline:
    def __init__(self, weights_path=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_resnet18_transfer(
            num_classes=len(CLASSES),
            freeze_backbone=False,
            weights_path=weights_path
        ).to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
        self.curator = BookLensNLPCurator()
        
    def _compute_sobel_density(self, img_pil):
        gray = np.array(img_pil.convert("L").resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255.0
        t_gray = torch.from_numpy(gray)[None, None]
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        gx = F.conv2d(t_gray, sobel_x, padding=1)[0, 0].numpy()
        gy = F.conv2d(t_gray, sobel_y, padding=1)[0, 0].numpy()
        mag = np.sqrt(gx**2 + gy**2)
        mag = mag / (mag.max() + 1e-8)
        return float((mag > 0.25).mean() * 100)

    def predict_and_curate(self, image_input, user_interest=None):
        if isinstance(image_input, str):
            img_pil = Image.open(image_input).convert("RGB")
        else:
            img_pil = image_input.convert("RGB")
        edge_density = self._compute_sobel_density(img_pil)
        tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            pred_idx = int(logits.argmax(1).item())
        pred_class = CLASSES[pred_idx]
        conf = float(probs[pred_idx])
        sorted_probs = np.sort(probs)[::-1]
        margin = float(sorted_probs[0] - sorted_probs[1])
        curation_text = self.curator.generate_curation(
            predicted_class=pred_class,
            confidence=conf,
            margin=margin,
            edge_density=edge_density,
            user_interest=user_interest
        )
        return {
            "predicted_class": pred_class,
            "predicted_name_kr": CLASS_KR[pred_class],
            "confidence": conf,
            "probabilities": {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))},
            "margin": margin,
            "edge_density": edge_density,
            "curation_report": curation_text
        }
