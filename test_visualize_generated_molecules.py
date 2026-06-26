import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem


REPO_ROOT = Path(__file__).resolve().parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "visualize_generated_molecules.py"


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


if __name__ == "__main__":
    unittest.main()
