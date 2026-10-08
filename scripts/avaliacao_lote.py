"""Varredura de angulos e niveis de ruido para ORB; escreve CSV.

Executar na raiz do projeto: python scripts/avaliacao_lote.py --image data/input/exemplo.png
"""
import argparse
import csv
import sys
from pathlib import Path
from statistics import mean
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orb_experimento import (create_orb, detect_and_describe, gaussian_noise,
                            ground_truth_precision, load_image, match_features,
                            rotate_image)


def main():
    parser = argparse.ArgumentParser(description="Experimentos de rotacao e ruido com ORB")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--angles", type=float, nargs="+", default=[0, 30, 60, 90, 120, 180])
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0, 10, 20])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--nfeatures", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path("data/output/avaliacao_lote.csv"))
    args = parser.parse_args()
    if args.repetitions <= 0 or args.nfeatures <= 0 or any(s < 0 for s in args.sigmas):
        parser.error("repetitions/nfeatures devem ser positivos e sigmas nao negativos")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    img = load_image(args.image)
    orb = create_orb(args.nfeatures)
    kp1, desc1, _ = detect_and_describe(orb, img)
    if desc1 is None:
        raise RuntimeError("Nenhum ponto encontrado na imagem de referencia")
    rows = []
    for angle in args.angles:
        rotated, transform = rotate_image(img, angle)
        for sigma in args.sigmas:
            for rep in range(args.repetitions):
                img2 = gaussian_noise(rotated, sigma, seed=42 + rep) if sigma else rotated
                kp2, desc2, orb_ms = detect_and_describe(orb, img2)
                started = perf_counter()
                matches = match_features(desc1, desc2)
                match_ms = (perf_counter() - started) * 1000
                gt = ground_truth_precision(kp1, kp2, matches, transform)
                rows.append({
                    "angulo_graus": angle,
                    "ruido_sigma": sigma,
                    "repeticao": rep + 1,
                    "keypoints_imagem1": len(kp1),
                    "keypoints_imagem2": len(kp2),
                    "matches_ratio": len(matches),
                    "acertos_geometricos": gt["acertos_geometricos"],
                    "precisao_pct": gt["precisao_geometrica_pct"],
                    "tempo_orb_imagem2_ms": round(orb_ms, 3),
                    "tempo_matching_ms": round(match_ms, 3),
                })
            subset = rows[-args.repetitions:]
            p = [r["precisao_pct"] for r in subset if r["precisao_pct"] is not None]
            print(f"angulo={angle:>5g}°, sigma={sigma:>4g}: "
                  f"matches medios={mean(r['matches_ratio'] for r in subset):.1f}, "
                  f"precisao media={mean(p):.1f}%" if p else
                  f"angulo={angle:g}°, sigma={sigma:g}: sem matches")

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV salvo em {args.output.resolve()}")


if __name__ == "__main__":
    main()
