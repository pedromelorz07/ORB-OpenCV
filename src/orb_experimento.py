"""Experimento didatico ORB (OpenCV): detectar, descrever e comparar imagens."""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import cv2 as cv
import numpy as np


def load_image(path: Path):
    image = cv.imread(str(path), cv.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Nao foi possivel abrir: {path}")
    return image


def create_orb(nfeatures: int):
    return cv.ORB_create(
        nfeatures=nfeatures,
        scaleFactor=1.2,
        nlevels=8,
        scoreType=cv.ORB_HARRIS_SCORE,
        WTA_K=2,
        fastThreshold=20,
    )


def detect_and_describe(orb, image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    start = perf_counter()
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    elapsed_ms = (perf_counter() - start) * 1000
    return keypoints, descriptors, elapsed_ms


def match_features(des1, des2, ratio=0.75):
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) < 2:
        return []
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    knn = matcher.knnMatch(des1, des2, k=2)
    good = [m for pair in knn if len(pair) == 2
            for m, n in [pair] if m.distance < ratio * n.distance]
    return sorted(good, key=lambda m: m.distance)


def rotate_image(image, angle):
    """Gira sem ampliar o canvas; a borda pode ficar cortada."""
    height, width = image.shape[:2]
    matrix = cv.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    rotated = cv.warpAffine(image, matrix, (width, height),
                            flags=cv.INTER_LINEAR,
                            borderMode=cv.BORDER_CONSTANT,
                            borderValue=(0, 0, 0))
    return rotated, matrix


def gaussian_noise(image, sigma, seed=42):
    noise = np.random.default_rng(seed).normal(0, sigma, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def ground_truth_precision(keypoints1, keypoints2, matches, matrix, threshold=3.0):
    """Precisao do matching em rotacao sintetica conhecida (erro <= threshold)."""
    if not matches:
        return {"acertos_geometricos": 0, "precisao_geometrica_pct": None}
    points = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    targets = np.float32([keypoints2[m.trainIdx].pt for m in matches])
    expected = points @ matrix[:, :2].T + matrix[:, 2]
    errors = np.linalg.norm(expected - targets, axis=1)
    correct = int(np.count_nonzero(errors <= threshold))
    return {
        "acertos_geometricos": correct,
        "precisao_geometrica_pct": round(100 * correct / len(matches), 2),
        "limiar_erro_px": threshold,
    }


def save_keypoints_csv(path, keypoints):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y", "size", "angle_degrees", "response", "octave"])
        for point in keypoints:
            writer.writerow([point.pt[0], point.pt[1], point.size,
                             point.angle, point.response, point.octave])


def save_detection(image, keypoints, name, output_dir):
    visual = cv.drawKeypoints(image, keypoints, None,
                              flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                              color=(0, 255, 0))
    cv.imwrite(str(output_dir / f"{name}_keypoints.png"), visual)
    save_keypoints_csv(output_dir / f"{name}_keypoints.csv", keypoints)


def main():
    parser = argparse.ArgumentParser(description="Teste de ORB no OpenCV")
    parser.add_argument("--image", type=Path, required=True,
                        help="Caminho da imagem de referencia")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--compare", type=Path, help="Segunda imagem real")
    group.add_argument("--rotate", type=float, help="Rotacao sintetica em graus")
    parser.add_argument("--noise", type=float, default=0.0,
                        help="Desvio-padrao do ruido gaussiano na imagem girada")
    parser.add_argument("--nfeatures", type=int, default=500)
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--output", type=Path, default=Path("data/output"))
    args = parser.parse_args()

    if args.nfeatures < 1:
        parser.error("--nfeatures deve ser positivo")
    if not 0 < args.ratio < 1:
        parser.error("--ratio deve estar entre 0 e 1")
    if args.noise < 0 or (args.noise > 0 and args.rotate is None):
        parser.error("--noise >= 0 e ruido positivo requer --rotate")

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    first = load_image(args.image)
    orb = create_orb(args.nfeatures)
    kp1, des1, time1 = detect_and_describe(orb, first)
    save_detection(first, kp1, "imagem1", output_dir)
    if des1 is not None:
        np.save(output_dir / "imagem1_descriptors.npy", des1)

    report = {
        "opencv_version": cv.__version__,
        "parametros": {"nfeatures": args.nfeatures, "WTA_K": 2,
                       "scoreType": "HARRIS", "ratio": args.ratio,
                       "rotation_degrees": args.rotate, "noise_sigma": args.noise},
        "imagem1": {"arquivo": str(args.image), "keypoints": len(kp1),
                    "descriptors_shape": list(des1.shape) if des1 is not None else None,
                    "tempo_orb_ms": round(time1, 3)},
    }

    second = None
    matrix = None
    if args.compare is not None:
        second = load_image(args.compare)
    elif args.rotate is not None:
        second, matrix = rotate_image(first, args.rotate)
        if args.noise:
            second = gaussian_noise(second, args.noise)
        cv.imwrite(str(output_dir / "imagem2_transformada.png"), second)

    if second is not None:
        kp2, des2, time2 = detect_and_describe(orb, second)
        save_detection(second, kp2, "imagem2", output_dir)
        if des2 is not None:
            np.save(output_dir / "imagem2_descriptors.npy", des2)
        start = perf_counter()
        matches = match_features(des1, des2, ratio=args.ratio)
        matching_ms = (perf_counter() - start) * 1000
        pair_report = {
            "keypoints": len(kp2),
            "descriptors_shape": list(des2.shape) if des2 is not None else None,
            "tempo_orb_ms": round(time2, 3),
            "correspondencias_apos_ratio": len(matches),
            "tempo_matching_ms": round(matching_ms, 3),
        }
        if matrix is not None:
            pair_report.update(ground_truth_precision(kp1, kp2, matches, matrix))
        # Homografia RANSAC: diagnostico; so e geometricamente apropriada
        # a cenas aproximadamente planas ou mudancas de ponto de vista sem paralaxe.
        if len(matches) >= 4:
            src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            H, mask = cv.findHomography(src, dst, cv.RANSAC, 3.0)
            if H is not None and mask is not None:
                pair_report["ransac_inliers"] = int(mask.sum())
                pair_report["ransac_inlier_ratio_pct"] = round(100 * mask.mean(), 2)
            else:
                pair_report["ransac_inliers"] = None
        else:
            pair_report["ransac_inliers"] = None
        report["imagem2_e_matching"] = pair_report
        canvas = cv.drawMatches(first, kp1, second, kp2, matches[:60], None,
                                flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv.imwrite(str(output_dir / "correspondencias.png"), canvas)

    with (output_dir / "resultados.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nArquivos salvos em: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
