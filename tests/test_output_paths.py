import tempfile
import unittest
from pathlib import Path

from remove_ai_watermarks._internal.output_paths import unique_output_path


class UniqueOutputPathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_free_path_is_returned_unchanged(self):
        desired = self.root / "photo.png"
        assert unique_output_path(desired) == desired
        assert not desired.exists()

    def test_existing_file_is_preserved(self):
        desired = self.root / "photo.png"
        desired.write_bytes(b"previous")

        result = unique_output_path(desired)

        assert result == self.root / "photo_1.png"
        assert desired.read_bytes() == b"previous"
        assert not result.exists()

    def test_multiple_collisions_are_skipped(self):
        for name in ("photo.png", "photo_1.png", "photo_2.png"):
            (self.root / name).write_bytes(name.encode())

        result = unique_output_path(self.root / "photo.png")

        assert result == self.root / "photo_3.png"

    def test_extension_is_preserved(self):
        desired = self.root / "clip.mp4"
        desired.write_bytes(b"previous")

        result = unique_output_path(desired)

        assert result.name == "clip_1.mp4"

    def test_broken_symlink_is_a_collision(self):
        desired = self.root / "photo.png"
        missing_target = self.root / "missing.png"

        try:
            desired.symlink_to(missing_target)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"Symbolic links unavailable: {exc}")

        result = unique_output_path(desired)

        assert result == self.root / "photo_1.png"
        assert desired.is_symlink()


if __name__ == "__main__":
    unittest.main()
