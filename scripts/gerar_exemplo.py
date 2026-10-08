"""Cria uma cena artificial com textura para testes sem fotos proprias."""
from pathlib import Path
import cv2 as cv
import numpy as np


def generate_scene(seed=1234):
    rng = np.random.default_rng(seed)
    image = np.full((600, 800, 3), (224, 227, 231), dtype=np.uint8)
    for _ in range(110):
        x, y = rng.integers(35, 760), rng.integers(35, 560)
        color = tuple(int(v) for v in rng.integers(20, 190, 3))
        shape = int(rng.integers(0, 3))
        if shape == 0:
            cv.circle(image, (int(x), int(y)), int(rng.integers(5, 20)), color, -1)
        elif shape == 1:
            cv.rectangle(image, (int(x), int(y)),
                         (int(x + rng.integers(8, 34)), int(y + rng.integers(8, 34))),
                         color, -1)
        else:
            cv.line(image, (int(x), int(y)),
                    (int(x + rng.integers(-30, 30)), int(y + rng.integers(-30, 30))),
                    color, 3)
    for j, label in enumerate(["ORB", "FAST", "HARRIS", "rBRIEF", "OpenCV"]):
        cv.putText(image, label, (30 + j * 140, 300),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (10, 20, 40), 2, cv.LINE_AA)
    return image


if __name__ == "__main__":
    output = Path("data/input/exemplo.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(output), generate_scene())
    print(f"Imagem criada: {output.resolve()}")
