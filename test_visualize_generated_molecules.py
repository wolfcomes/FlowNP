import subprocess
import sys
import tempfile
import unittest
import csv
import importlib.util
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem


REPO_ROOT = Path(__file__).resolve().parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "visualize_generated_molecules.py"

spec = importlib.util.spec_from_file_location("visualize_generated_molecules", SCRIPT_PATH)
visualize_generated_molecules = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(visualize_generated_molecules)


def write_input_sdf(path: Path, n_molecules: int = 25) -> None:
    smiles_list = [
        "CCO",
        "CCN",
        "CCC",
        "c1ccccc1",
        "CC(=O)O",
    ]
    writer = Chem.SDWriter(str(path))
    for idx in range(n_molecules):
        mol = Chem.MolFromSmiles(smiles_list[idx % len(smiles_list)])
        mol.SetProp("_Name", f"mol_{idx}")
        AllChem.Compute2DCoords(mol)
        writer.write(mol)
    writer.close()


class VisualizeGeneratedMoleculesCliTests(unittest.TestCase):
    def test_prepare_molecule_for_drawing_preserves_stereochemistry(self) -> None:
        mol = Chem.MolFromSmiles("C[C@H](O)F")

        drawable = visualize_generated_molecules.prepare_molecule_for_drawing(mol)

        self.assertIn("@", Chem.MolToSmiles(drawable, isomericSmiles=True))

    def test_cli_writes_grid_image_and_selected_sdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_sdf = tmp_path / "generated.sdf"
            output_png = tmp_path / "selected_grid.png"
            output_sdf = tmp_path / "selected.sdf"
            write_input_sdf(input_sdf)

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--input",
                    str(input_sdf),
                    "--output",
                    str(output_png),
                    "--output-sdf",
                    str(output_sdf),
                    "--count",
                    "20",
                ],
                cwd=tmp_path,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)
            self.assertTrue(output_png.is_file())
            self.assertGreater(output_png.stat().st_size, 0)
            self.assertTrue(output_sdf.is_file())

            selected = [mol for mol in Chem.SDMolSupplier(str(output_sdf), removeHs=False) if mol is not None]
            self.assertEqual(len(selected), 20)
            self.assertIn("Selected 20 molecules", result.stdout)

    def test_cli_visualizes_molecules_from_csv_smiles_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_csv = tmp_path / "generated.csv"
            output_png = tmp_path / "selected_grid.png"
            output_sdf = tmp_path / "selected.sdf"

            smiles_values = ["CCO", "CCN", "CCC", "c1ccccc1", "CC(=O)O"] * 5
            with input_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["Name", "smiles", "Source"])
                writer.writeheader()
                for idx, smiles in enumerate(smiles_values):
                    writer.writerow({"Name": f"mol_{idx}", "smiles": smiles, "Source": "test"})

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--input-csv",
                    str(input_csv),
                    "--smiles-column",
                    "smiles",
                    "--output",
                    str(output_png),
                    "--output-sdf",
                    str(output_sdf),
                    "--count",
                    "20",
                ],
                cwd=tmp_path,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)
            self.assertTrue(output_png.is_file())
            self.assertGreater(output_png.stat().st_size, 0)

            selected = [mol for mol in Chem.SDMolSupplier(str(output_sdf), removeHs=False) if mol is not None]
            self.assertEqual(len(selected), 20)
            self.assertIn("Selected 20 molecules", result.stdout)


if __name__ == "__main__":
    unittest.main()
