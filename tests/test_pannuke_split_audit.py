import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_pannuke_splits import image_digest


class PanNukeSplitAuditTest(unittest.TestCase):
    def test_digest_is_content_sensitive_and_layout_independent(self):
        image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        self.assertEqual(image_digest(image), image_digest(image.copy()))
        changed = image.copy()
        changed[0, 0, 0] += 1
        self.assertNotEqual(image_digest(image), image_digest(changed))


if __name__ == "__main__":
    unittest.main()
