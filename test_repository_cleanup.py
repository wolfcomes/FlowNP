import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SCAN_ROOTS = [
    REPO_ROOT / "src",
    REPO_ROOT / "configs",
]
TOP_LEVEL_PATTERNS = ["*.py"]
REMOVED_TOKENS = [
    "self_conditioning",
    "self-conditioning",
    "SelfConditioning",
    "prev_dst_dict",
]


class RepositoryCleanupTests(unittest.TestCase):
    def test_self_conditioning_code_and_config_are_removed(self) -> None:
        matches: list[str] = []

        for path in self._iter_scanned_files():
            text = path.read_text(encoding="utf-8", errors="ignore")
            for token in REMOVED_TOKENS:
                if token in text:
                    relative_path = path.relative_to(REPO_ROOT)
                    matches.append(f"{relative_path}: contains {token!r}")

        self.assertEqual(matches, [], "\n".join(matches))

    def _iter_scanned_files(self):
        for root in SCAN_ROOTS:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if path.is_file() and path.suffix in {".py", ".yaml", ".yml"}:
                    yield path

        for pattern in TOP_LEVEL_PATTERNS:
            for path in REPO_ROOT.glob(pattern):
                if path.name != Path(__file__).name:
                    yield path


if __name__ == "__main__":
    unittest.main()
