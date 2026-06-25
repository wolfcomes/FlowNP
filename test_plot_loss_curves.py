import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "plot_loss_curves.py"


def write_experiment(root: Path, exp_name: str, knn_k: int, train_values: list[float], val_values: list[float]) -> None:
    exp_dir = root / exp_name / "csv_logs" / "version_0"
    exp_dir.mkdir(parents=True, exist_ok=True)

    hparams_path = exp_dir / "hparams.yaml"
    hparams_path.write_text(
        "\n".join(
            [
                "vector_field_config:",
                f"  knn_connectivity: {knn_k}",
                "  enable_dynamic_graph: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    metrics_path = exp_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "epoch_exact",
                "step",
                "train_total_loss",
                "val_total_loss_epoch",
            ]
        )
        for idx, (train_loss, val_loss) in enumerate(zip(train_values, val_values)):
            writer.writerow([idx, float(idx), idx * 100, train_loss, val_loss])


class PlotLossCurvesCliTests(unittest.TestCase):
    def test_knn_root_generates_paper_outputs_and_recommends_plateau_k(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            knn_root = tmp_path / "checkpoints_coconut"

            experiments = {
                "SDE-k=8-v14": (
                    8,
                    [9.2, 7.8, 6.1, 5.0, 4.1, 3.6, 3.2, 2.95, 2.83, 2.74],
                    [10.0, 8.0, 6.0, 5.0, 4.0, 3.5, 3.2, 3.0, 2.95, 2.90],
                ),
                "SDE-k=12-v13": (
                    12,
                    [8.8, 6.8, 5.0, 4.0, 3.1, 2.6, 2.25, 2.05, 1.97, 1.92],
                    [9.5, 7.0, 5.0, 4.0, 3.0, 2.5, 2.2, 2.0, 1.92, 1.90],
                ),
                "SDE-k=16-v15": (
                    16,
                    [8.5, 6.2, 4.5, 3.4, 2.7, 2.25, 2.02, 1.90, 1.86, 1.83],
                    [9.0, 6.5, 4.5, 3.2, 2.5, 2.1, 1.95, 1.87, 1.84, 1.82],
                ),
                "SDE-k=24-v16": (
                    24,
                    [8.35, 6.0, 4.25, 3.2, 2.58, 2.12, 1.96, 1.87, 1.82, 1.80],
                    [8.9, 6.3, 4.3, 3.0, 2.4, 2.0, 1.90, 1.85, 1.81, 1.79],
                ),
                "SDE-k=32-v17": (
                    32,
                    [8.3, 5.95, 4.2, 3.15, 2.55, 2.08, 1.93, 1.85, 1.80, 1.79],
                    [8.85, 6.25, 4.25, 2.98, 2.38, 1.98, 1.88, 1.84, 1.80, 1.78],
                ),
            }

            for exp_name, (knn_k, train_values, val_values) in experiments.items():
                write_experiment(knn_root, exp_name, knn_k, train_values, val_values)

            result = subprocess.run(
                [sys.executable, str(SCRIPT_PATH), "--knn-root", str(knn_root)],
                cwd=tmp_path,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)

            output_dir = tmp_path / "evaluation_results" / "loss_visiualization"
            self.assertTrue((output_dir / "knn_loss_main.png").is_file())
            self.assertTrue((output_dir / "knn_loss_train_val.png").is_file())
            self.assertTrue((output_dir / "knn_loss_summary.csv").is_file())
            self.assertTrue((output_dir / "knn_recommendation.txt").is_file())

            with (output_dir / "knn_loss_summary.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual([int(row["knn_connectivity"]) for row in rows], [8, 12, 16, 24, 32])
            plateau_rows = [row for row in rows if row["is_plateau_candidate"] == "True"]
            self.assertTrue(any(int(row["knn_connectivity"]) == 24 for row in plateau_rows))

            recommendation_text = (output_dir / "knn_recommendation.txt").read_text(encoding="utf-8")
            self.assertIn("Recommended K: 24", recommendation_text)
            self.assertIn("relative_improvement_vs_prev_k", recommendation_text)


if __name__ == "__main__":
    unittest.main()
