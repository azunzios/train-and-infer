"""Exercise detection coverage, threaded decoding, CLI and adaptive scan behavior."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from ved.inference import Detector
from ved.model import EventClassifier, load_export, save_export


class DetectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.config = dict(img_size=16, patch_size=16, tubelet_size=1, num_frames=4,
                          embed_dim=64, depth=1, num_heads=1, drop_path_rate=0)
        model = EventClassifier(cls.config, 2, head_dropout=0).eval()
        cls.checkpoint = cls.root / "model.pt"
        save_export(model, str(cls.checkpoint), {
            "class_names": ["motion_b", "motion_a"],
            "display_names": {"motion_b": "Motion B", "motion_a": "Motion A"},
            "thresholds": {"motion_b": 0.7, "motion_a": 0.8},
            "fps": 4, "num_frames": 4, "img_size": 16,
        })
        cls.video = cls.make_video("source.mp4", 96)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def make_video(cls, name, count):
        path = cls.root / name
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8, (32, 32))
        if not writer.isOpened():
            raise RuntimeError("Cannot create test video")
        try:
            for index in range(count):
                value = 240 if 40 <= index < 56 else 20
                writer.write(np.full((32, 32, 3), value, dtype=np.uint8))
        finally:
            writer.release()
        return path

    def detector(self):
        return Detector(str(self.checkpoint), device="cpu")

    def test_export_roundtrip_names_and_threshold_aliases(self):
        _, meta = load_export(str(self.checkpoint))
        self.assertEqual(meta["class_names"], ["motion_b", "motion_a"])
        self.assertEqual(meta["thresholds"], {"motion_b": 0.7, "motion_a": 0.8})
        detector = self.detector()
        self.assertEqual(detector.resolve_thresholds({"event_2": 0.2, "motion_b": 0.4}),
                         {"motion_b": 0.4, "motion_a": 0.2})
        with self.assertRaises(ValueError):
            detector.resolve_thresholds({"event_1": 0.4, "motion_b": 0.2})
        renamed = Detector(str(self.checkpoint), device="cpu", class_names=["b", "a"])
        self.assertEqual(renamed.thresholds, {"b": 0.7, "a": 0.8})

    def test_prefetch_preserves_probabilities_and_tail_coverage(self):
        detector = self.detector()
        sync = detector.predict_windows(str(self.video), 0.75, 3, False, False)
        threaded = detector.predict_windows(str(self.video), 0.75, 3, False, True)
        np.testing.assert_array_equal(sync[0], threaded[0])
        np.testing.assert_allclose(sync[1], threaded[1], atol=1e-6)
        self.assertEqual(sync[2], 48)
        self.assertEqual(sync[0][-1], 48 - detector.win)
        short = self.make_video("short.mp4", 2)
        starts, probs, count = detector.predict_windows(str(short), 0.5, 2, False)
        self.assertEqual(count, 1)
        self.assertEqual(starts.tolist(), [0])
        self.assertEqual(probs.shape, (1, 2))

    def test_manual_range_has_original_timestamps_and_no_outside_detections(self):
        detector = self.detector()
        result = detector.detect(str(self.video), start_sec=3, end_sec=6,
                                 thresholds={"event_1": 0, "event_2": 0},
                                 smooth_sec=2, verbose=False)
        self.assertTrue(result["segments"])
        for segment in result["segments"]:
            self.assertEqual((segment.start, segment.end), (3, 6))
        self.assertEqual(result["performance"]["analyzed_range"], [3, 6])
        np.testing.assert_array_equal(result["frame_scores"][:12], 0)

    def test_adaptive_reduces_windows_on_background_and_refines_candidates(self):
        detector = self.detector()
        with patch.object(detector, "_run", side_effect=lambda windows: np.full((len(windows), 2), 0.01)):
            full = detector.detect(str(self.video), stride_sec=0.25, verbose=False)
            adaptive = detector.detect(str(self.video), stride_sec=0.25, scan="adaptive", verbose=False)
        self.assertFalse(adaptive["segments"])
        self.assertEqual(adaptive["performance"]["refined_ranges"], [])
        self.assertLess(adaptive["performance"]["windows"], full["performance"]["windows"] / 2)

        def brightness(windows):
            return np.array([[0.95, 0.95] if window.max() > 180 else [0.01, 0.01]
                             for window in windows], dtype=np.float32)

        with patch.object(detector, "_run", side_effect=brightness):
            result = detector.detect(str(self.video), stride_sec=0.25, scan="adaptive",
                                     refine_padding=0, smooth_sec=0, merge_gap=0,
                                     min_dur=0.1, verbose=False)
        self.assertTrue(result["performance"]["refined_ranges"])
        self.assertTrue(result["segments"])
        self.assertTrue(all(segment.start < 7 and segment.end > 5 for segment in result["segments"]))
        lo, hi = result["performance"]["refined_ranges"][0]
        self.assertGreater(lo, 0)
        self.assertLess(hi, 12)

    def test_decoder_closes_when_model_fails_during_prefetch(self):
        detector = self.detector()
        closed = []

        def frames(*args):
            try:
                for _ in range(100):
                    yield np.zeros((16, 16, 3), np.uint8)
            finally:
                closed.append(True)

        with patch("ved.inference.iter_frames", frames), patch.object(detector, "_run", side_effect=RuntimeError("model failed")):
            with self.assertRaisesRegex(RuntimeError, "model failed"):
                detector.predict_windows(str(self.video), 0.5, 2, False, True)
        self.assertEqual(closed, [True])

    def test_local_cli_outputs_original_labels_without_zip(self):
        from infer import main

        output = self.root / "cli"
        with redirect_stdout(io.StringIO()) as log:
            main(["--video", str(self.video), "--ckpt", str(self.checkpoint),
                  "--output", str(output), "--device", "cpu", "--profile", "fast",
                  "--start-time", "3", "--end-time", "6", "--thr", "event_1=0", "event_2=0",
                  "--min-dur", "0.1", "--no-plot", "--quiet"])
        result = json.loads((output / "source.json").read_text())
        self.assertEqual(set(result["thresholds"]), {"motion_b", "motion_a"})
        self.assertEqual(result["performance"]["stride_sec"], 1)
        self.assertEqual(result["performance"]["batch_size"], 16)
        self.assertTrue(Path(result["best_clip"]["file"]).is_file())
        self.assertIn("Motion B", log.getvalue())
        self.assertFalse(list(output.glob("*.zip")))

    def test_no_detection_does_not_export_clip(self):
        from ved.outputs import write_outputs

        detector = self.detector()
        result = detector.detect(str(self.video), thresholds={"event_1": 1, "event_2": 1}, verbose=False)
        output = self.root / "no-events"
        paths = write_outputs(result, detector, output, plot=False)
        self.assertNotIn("clip", paths)
        self.assertIsNone(json.loads(paths["json"].read_text())["best_clip"])


if __name__ == "__main__":
    unittest.main()
