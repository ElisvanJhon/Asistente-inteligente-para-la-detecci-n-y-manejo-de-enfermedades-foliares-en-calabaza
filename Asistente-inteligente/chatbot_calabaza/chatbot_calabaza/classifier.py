"""
Wrapper simple sobre el modelo YOLOv8n-cls entrenado en Colab.
"""

from ultralytics import YOLO


class LeafClassifier:
    def __init__(self, weights_path: str = "models/best.pt"):
        self.model = YOLO(weights_path)

    def predict(self, image_path: str) -> dict:
        """
        Devuelve: {"clase": str, "confianza": float, "top5": list[tuple[str, float]]}
        """
        results = self.model.predict(image_path, verbose=False)
        r = results[0]

        probs = r.probs
        top1_idx = int(probs.top1)
        top1_conf = float(probs.top1conf)
        clase = r.names[top1_idx]

        top5_idx = probs.top5
        top5_conf = probs.top5conf.tolist()
        top5 = [(r.names[i], c) for i, c in zip(top5_idx, top5_conf)]

        return {"clase": clase, "confianza": top1_conf, "top5": top5}
