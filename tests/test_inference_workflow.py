"""Regression checks for checkpoint compatibility and URL inference artifacts."""
from __future__ import annotations

from contextlib import redirect_stdout
from functools import partial
import http.server
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
import zipfile

from ved.labels import neutral_metadata
from ved.media import parse_time, time_range


class MetadataTests(unittest.TestCase):
    def test_order_and_thresholds_are_preserved_without_legacy_text(self):
        original = {
            "class_names": ["legacy_second", "legacy_first"],
            "display_names": {"legacy_second": "Old label"},
            "thresholds": {"legacy_first": 0.72, "legacy_second": 0.34},
            "fps": 8, "num_frames": 16, "img_size": 224,
            "cfg": {"class_names": ["legacy_second", "legacy_first"]},
        }
        metadata = neutral_metadata(original, 2)
        self.assertEqual(metadata["class_names"], ["event_1", "event_2"])
        self.assertEqual(metadata["thresholds"], {"event_1": 0.34, "event_2": 0.72})
        self.assertEqual(metadata["display_names"], {"event_1": "Event 1", "event_2": "Event 2"})
        self.assertNotIn("legacy", json.dumps(metadata))
        self.assertEqual(neutral_metadata(metadata, 2), metadata)
        self.assertIn("cfg", original)

    def test_inconsistent_class_count_is_rejected(self):
        with self.assertRaises(ValueError):
            neutral_metadata({"class_names": ["first"]}, 2)

    def test_relabeling_preserves_detected_intervals_and_scores(self):
        import numpy as np
        from ved.postprocess import frame_scores_to_segments

        metadata = {
            "class_names": ["previous_b", "previous_a"],
            "thresholds": {"previous_a": 0.66, "previous_b": 0.37},
            "fps": 8, "num_frames": 16, "img_size": 224,
        }
        neutral = neutral_metadata(metadata, 2)
        scores = np.full((80, 2), 0.1, dtype=np.float32)
        scores[8:32, 0] = 0.85
        scores[40:72, 1] = 0.95
        before = frame_scores_to_segments(scores, 8, metadata["class_names"], metadata["thresholds"])
        after = frame_scores_to_segments(scores, 8, neutral["class_names"], neutral["thresholds"])
        self.assertTrue(before)
        self.assertEqual(len(before), len(after))
        for original, renamed in zip(before, after):
            expected = original.to_dict()
            expected["label"] = neutral["class_names"][metadata["class_names"].index(original.label)]
            self.assertEqual(expected, renamed.to_dict())


class TimeTests(unittest.TestCase):
    def test_supported_time_formats(self):
        self.assertEqual(parse_time("12.5"), 12.5)
        self.assertEqual(parse_time("01:02.5"), 62.5)
        self.assertEqual(parse_time("01:02:03.5"), 3723.5)
        self.assertEqual(time_range("60", "01:30"), (60, 90))
        self.assertEqual(time_range("00:01:00", ""), (60, None))

    def test_invalid_ranges_fail_before_download(self):
        for value in ("-1", "nan", "inf", "00:60", "01::02", "1:2:3:4", "1.5:00"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_time(value)
        for start, end in (("5", "4"), ("5", "5")):
            with self.assertRaises(ValueError):
                time_range(start, end)


class NotebookTests(unittest.TestCase):
    def test_inference_notebook_has_valid_python_and_form_fields(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "notebooks/infer_colab.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(len(notebook["cells"]), 1)
        self.assertEqual(notebook["cells"][0]["cell_type"], "code")
        all_code = ""
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                compile(source, f"infer_colab:{index}", "exec")
                self.assertEqual(cell["outputs"], [])
                all_code += source
        for name in ("video_url", "output_dir", "batch_size", "local_checkpoint"):
            self.assertRegex(all_code, rf'{name} = .*# @param')
        self.assertNotIn("train_model", all_code)
        for obsolete in ("yt_dlp", "ffmpeg", "deno", "from infer_url", "start_time =", "end_time ="):
            self.assertNotIn(obsolete, all_code)
        self.assertIn('["wget", "-O",', all_code)
        self.assertIn('"effective_thresholds": effective_thresholds', all_code)
        self.assertIn('arcname="run_config.json"', all_code)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
class PipelineTests(unittest.TestCase):
    def test_download_interval_inference_and_archive(self):
        import torch

        from infer_url import run_pipeline
        from ved.model import EventClassifier, load_export

        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                "-i", "testsrc2=size=64x64:rate=8:duration=4", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", str(source),
            ], check=True)
            config = dict(img_size=16, patch_size=16, tubelet_size=1, num_frames=2,
                          embed_dim=64, depth=1, num_heads=1, drop_path_rate=0)
            model = EventClassifier(config, 2, head_dropout=0).eval()
            checkpoint = root / "model.pt"
            # Simulate an existing export with arbitrary historical class names.
            torch.save({
                "state_dict": model.state_dict(), "backbone_cfg": config, "num_classes": 2,
                "meta": {"class_names": ["previous_a", "previous_b"],
                         "thresholds": {"previous_a": 0.31, "previous_b": 0.79},
                         "display_names": {"previous_a": "Previous A"},
                         "cfg": {"label": "previous_a"},
                         "fps": 4, "num_frames": 2, "img_size": 16},
            }, checkpoint)
            loaded, metadata = load_export(str(checkpoint))
            sample = torch.rand(1, 3, 2, 16, 16)
            with torch.no_grad():
                torch.testing.assert_close(loaded(sample), model(sample))
            self.assertEqual(metadata["thresholds"], {"event_1": 0.31, "event_2": 0.79})

            class QuietHandler(http.server.SimpleHTTPRequestHandler):
                def log_message(self, *args):
                    pass

            handler = partial(QuietHandler, directory=str(root))
            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                with redirect_stdout(io.StringIO()) as log:
                    paths = run_pipeline(
                        f"http://127.0.0.1:{server.server_port}/source.mp4",
                        output=root / "results", start_time="1", end_time="3",
                        ckpt=str(checkpoint), batch_size=2, stride=0.25,
                        thresholds={"event_1": 0.0, "event_2": 0.0},
                        min_duration=0.1, smooth=0, merge_gap=0, device="cpu",
                    )
            finally:
                server.shutdown()
                server.server_close()
                worker.join(timeout=5)
            for path in paths.values():
                self.assertGreater(path.stat().st_size, 0)
            probe = subprocess.run([
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(paths["video"]),
            ], check=True, capture_output=True, text=True)
            self.assertAlmostEqual(float(probe.stdout.strip()), 2, delta=0.3)
            result = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertEqual(result["source_range"], {"start": 1, "end": 3})
            self.assertEqual(set(result["thresholds"]), {"event_1", "event_2"})
            self.assertEqual({s["label"] for s in result["segments"]}, {"event_1", "event_2"})
            for segment in result["segments"]:
                self.assertAlmostEqual(segment["source_start"], segment["start"] + 1)
                self.assertAlmostEqual(segment["source_end"], segment["end"] + 1)
            self.assertNotIn("previous_", json.dumps(result) + log.getvalue())
            with zipfile.ZipFile(paths["archive"]) as bundle:
                self.assertEqual(set(bundle.namelist()), {"clip.mp4", "result.json", "timeline.png"})


if __name__ == "__main__":
    unittest.main()
