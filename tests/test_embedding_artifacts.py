import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from embedding_artifacts import FEATURE_DIM, publish_batches, save_batch


class PublishBatchesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.images = np.arange(6 * 2 * 2 * 3, dtype=np.uint8).reshape(6, 2, 2, 3)
        self.labels = np.asarray(["a", "b", "c", "d", "e", "f"])
        self.image_path = self.root / "images.npy"
        self.label_path = self.root / "types.npy"
        np.save(self.image_path, self.images)
        np.save(self.label_path, self.labels)

    def tearDown(self):
        self.temp.cleanup()

    def _features(self, start, end):
        values = np.arange(start, end, dtype=np.float32)[:, None]
        return np.broadcast_to(values, (end - start, FEATURE_DIM)).copy()

    def _batch(self, start, end):
        save_batch(
            self.root / f"embeddings_batch_{start}_{end}.npz",
            start,
            self._features(start, end),
            self.images[start:end],
        )

    def test_exact_coverage_publishes_in_index_order(self):
        self._batch(3, 6)
        self._batch(0, 3)
        self.assertEqual(publish_batches(self.root, self.image_path, self.label_path), (6, FEATURE_DIM))
        self.assertTrue(np.array_equal(np.load(self.root / "labels.npy"), self.labels))
        self.assertTrue(np.array_equal(np.load(self.root / "embeddings.npy")[:, 0], np.arange(6)))

    def test_gap_is_rejected_without_publication(self):
        self._batch(0, 2)
        self._batch(3, 6)
        with self.assertRaisesRegex(ValueError, "missing"):
            publish_batches(self.root, self.image_path, self.label_path)
        self.assertFalse((self.root / "embeddings.npy").exists())

    def test_overlap_is_rejected(self):
        self._batch(0, 4)
        self._batch(3, 6)
        with self.assertRaisesRegex(ValueError, "overlapping"):
            publish_batches(self.root, self.image_path, self.label_path)

    def test_source_image_change_is_rejected(self):
        self._batch(0, 6)
        changed = self.images.copy()
        changed[2, 0, 0, 0] += 1
        np.save(self.image_path, changed)
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            publish_batches(self.root, self.image_path, self.label_path)


if __name__ == "__main__":
    unittest.main()
