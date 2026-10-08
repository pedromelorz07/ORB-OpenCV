"""Testes de integridade simples do pipeline ORB."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from orb_experimento import create_orb, detect_and_describe, match_features, rotate_image, ground_truth_precision
from gerar_exemplo import generate_scene


class TestORB(unittest.TestCase):
    def setUp(self):
        self.image = generate_scene()
        self.orb = create_orb(500)

    def test_detection_and_binary_descriptors(self):
        keypoints, desc, elapsed = detect_and_describe(self.orb, self.image)
        self.assertGreater(len(keypoints), 20)
        self.assertEqual(desc.shape, (len(keypoints), 32))
        self.assertEqual(desc.dtype.name, 'uint8')
        self.assertGreaterEqual(elapsed, 0)

    def test_matching_rotation(self):
        rotated, M = rotate_image(self.image, 30)
        kp1, d1, _ = detect_and_describe(self.orb, self.image)
        kp2, d2, _ = detect_and_describe(self.orb, rotated)
        matches = match_features(d1, d2)
        self.assertGreater(len(matches), 5)
        result = ground_truth_precision(kp1, kp2, matches, M)
        self.assertGreater(result["acertos_geometricos"], 0)


if __name__ == "__main__":
    unittest.main()
