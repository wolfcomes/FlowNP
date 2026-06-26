from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Sequence

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw


DEFAULT_OUTPUT = "evaluation_results/molecule_visualization/generated_molecules_grid.png"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def load_valid_molecules_from_sdf(input_path: Path) -> list[Chem.Mol]:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input SDF file not found: {input_path}")

    supplier = Chem.SDMolSupplier(str(input_path), removeHs=False)
    molecules: list[Chem.Mol] = []
    for mol in supplier:
        if mol is None:
            continue
        molecules.append(mol)

    if not molecules:
        raise ValueError(f"No valid molecules were found in {input_path}")

    return molecules


def load_valid_molecules_from_csv(input_path: Path, smiles_column: str, name_column: str | None) -> list[Chem.Mol]:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV file not found: {input_path}")

    molecules: list[Chem.Mol] = []
    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file is missing a header row: {input_path}")
        if smiles_column not in reader.fieldnames:
            available = ", ".join(reader.fieldnames)
            raise ValueError(f"Column '{smiles_column}' was not found in {input_path}. Available columns: {available}")

        for row_idx, row in enumerate(reader, start=1):
            smiles = (row.get(smiles_column) or "").strip()
            if not smiles:
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            if name_column and row.get(name_column):
                mol.SetProp("_Name", row[name_column].strip())
            elif row.get("Name"):
                mol.SetProp("_Name", row["Name"].strip())
            else:
                mol.SetProp("_Name", f"row_{row_idx}")
            molecules.append(mol)

    if not molecules:
        raise ValueError(f"No valid SMILES were found in {input_path}")

    return molecules


def configure_rdkit_logging(show_warnings: bool) -> None:
    if show_warnings:
        RDLogger.EnableLog("rdApp.warning")
        RDLogger.EnableLog("rdApp.error")
    else:
        RDLogger.DisableLog("rdApp.warning")
        RDLogger.DisableLog("rdApp.error")


def select_molecules(
    molecules: Sequence[Chem.Mol],
    count: int,
    start: int,
    random_sample: bool,
    seed: int,
) -> list[Chem.Mol]:
    if len(molecules) < count:
        raise ValueError(f"Requested {count} molecules, but only {len(molecules)} valid molecules were found.")

    if random_sample:
        rng = random.Random(seed)
        return list(rng.sample(list(molecules), count))

    end = start + count
    if end > len(molecules):
        raise ValueError(
            f"Requested molecules [{start}:{end}], but only {len(molecules)} valid molecules were found."
        )
    return list(molecules[start:end])


def prepare_molecule_for_drawing(mol: Chem.Mol) -> Chem.Mol:
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    drawable = Chem.MolFromSmiles(smiles)
    if drawable is None:
        drawable = Chem.Mol(mol)
    AllChem.Compute2DCoords(drawable)
    if mol.HasProp("_Name"):
        drawable.SetProp("_Name", mol.GetProp("_Name"))
    return drawable


def legend_for_molecule(mol: Chem.Mol, index: int, mode: str) -> str:
    if mode == "none":
        return ""
    if mode == "name":
        if mol.HasProp("_Name") and mol.GetProp("_Name").strip():
            return mol.GetProp("_Name").strip()
        return f"Mol {index + 1}"
    if mode == "smiles":
        return Chem.MolToSmiles(mol)
    return f"Mol {index + 1}"


def write_selected_sdf(molecules: Sequence[Chem.Mol], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(output_path))
    for mol in molecules:
        writer.write(mol)
    writer.close()


def draw_molecule_grid(
    molecules: Sequence[Chem.Mol],
    output_path: Path,
    mols_per_row: int,
    sub_img_size: int,
    legend_mode: str,
    preserve_coordinates: bool,
) -> None:
    if preserve_coordinates:
        drawable_mols = [Chem.Mol(mol) for mol in molecules]
    else:
        drawable_mols = [prepare_molecule_for_drawing(mol) for mol in molecules]
    legends = [legend_for_molecule(mol, idx, legend_mode) for idx, mol in enumerate(drawable_mols)]
    image = Draw.MolsToGridImage(
        drawable_mols,
        molsPerRow=mols_per_row,
        subImgSize=(sub_img_size, sub_img_size),
        legends=legends,
        useSVG=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(output_path))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select generated molecules from an SDF file and save a 2D visualization grid."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        help="Input SDF file containing generated molecules.",
    )
    input_group.add_argument(
        "--input-csv",
        help="Input CSV file containing generated molecule SMILES.",
    )
    parser.add_argument(
        "--smiles-column",
        default="smiles",
        help="SMILES column name when using --input-csv. Defaults to smiles.",
    )
    parser.add_argument(
        "--name-column",
        default="Name",
        help="Optional molecule name column when using --input-csv. Defaults to Name.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output PNG path. Defaults to {DEFAULT_OUTPUT}.",
    )
    parser.add_argument(
        "--output-sdf",
        default=None,
        help="Optional SDF path for saving the selected molecules.",
    )
    parser.add_argument(
        "--count",
        type=positive_int,
        default=20,
        help="Number of valid molecules to visualize. Defaults to 20.",
    )
    parser.add_argument(
        "--start",
        type=non_negative_int,
        default=0,
        help="Start index for sequential selection. Defaults to 0.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Randomly sample molecules instead of selecting sequentially.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used with --random. Defaults to 0.",
    )
    parser.add_argument(
        "--mols-per-row",
        type=positive_int,
        default=5,
        help="Number of molecules per grid row. Defaults to 5.",
    )
    parser.add_argument(
        "--sub-img-size",
        type=positive_int,
        default=260,
        help="Pixel size for each molecule panel. Defaults to 260.",
    )
    parser.add_argument(
        "--legend",
        choices=("index", "name", "smiles", "none"),
        default="index",
        help="Legend style below each molecule. Defaults to index.",
    )
    parser.add_argument(
        "--show-rdkit-warnings",
        action="store_true",
        help="Show RDKit warnings for invalid molecules while reading the input SDF.",
    )
    parser.add_argument(
        "--preserve-sdf-coordinates",
        action="store_true",
        help="Use coordinates stored in the SDF instead of regenerating clean 2D depictions.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input or args.input_csv).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_sdf = Path(args.output_sdf).expanduser().resolve() if args.output_sdf else None

    configure_rdkit_logging(args.show_rdkit_warnings)
    if args.input_csv:
        name_column = args.name_column if args.name_column else None
        molecules = load_valid_molecules_from_csv(input_path, args.smiles_column, name_column)
    else:
        molecules = load_valid_molecules_from_sdf(input_path)
    selected = select_molecules(
        molecules=molecules,
        count=args.count,
        start=args.start,
        random_sample=args.random,
        seed=args.seed,
    )

    draw_molecule_grid(
        molecules=selected,
        output_path=output_path,
        mols_per_row=args.mols_per_row,
        sub_img_size=args.sub_img_size,
        legend_mode=args.legend,
        preserve_coordinates=args.preserve_sdf_coordinates,
    )

    if output_sdf is not None:
        write_selected_sdf(selected, output_sdf)

    print(f"Selected {len(selected)} molecules from {input_path}")
    print(f"Saved visualization grid to {output_path}")
    if output_sdf is not None:
        print(f"Saved selected molecules to {output_sdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
