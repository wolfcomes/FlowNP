import glob
import os
import argparse
from rdkit import Chem
from rdkit.Chem import rdmolops
from oddt.toolkits.extras.rdkit import fixer
from openbabel import pybel

def split_pocket_ligand(input_path, protonate=False, radius=20):
    root = os.path.dirname(input_path)
    pdb_code = os.path.basename(input_path)[:4] + f'_cut{radius}'
    root = os.path.join(root, pdb_code)
    os.makedirs(root, exist_ok=True)
    complex_ = Chem.MolFromPDBFile(input_path, sanitize=False)
    try:
        pocket, ligand = fixer.ExtractPocketAndLigand(complex_, cutoff=radius)
        
        # Write pocket PDB file

        pocket_pdb_file = os.path.join(root, f"{pdb_code}_pocket.pdb")

        Chem.MolToPDBFile(pocket, pocket_pdb_file)

        

        if protonate:

            # Remove hydrogen and protonation

            inter_pdb_file = os.path.join(root, f"{pdb_code}_pocket_withH.pdb")

            Chem.MolToPDBFile(pocket, inter_pdb_file)

            os.system(f"obabel {inter_pdb_file} -d -O {pocket_pdb_file}")

            os.unlink(inter_pdb_file)
        
        # Write ligand PDB and SDF files
        ligand_pdb_file = os.path.join(root, f"{pdb_code}_ligand.pdb")
        Chem.MolToPDBFile(ligand, ligand_pdb_file)
        mol = next(pybel.readfile('pdb', ligand_pdb_file))
        mol.write('sdf', os.path.join(root, f"{pdb_code}_ligand.sdf"), overwrite=True)

    except Exception as e:
        print(f'Error: {e}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split pocket and ligand from a PDB file.')
    parser.add_argument('--input', type=str, help='Inp PDB file path')
    parser.add_argument('--protonate', action='store_true', help='Protonate pocket after extracting')
    parser.add_argument('--radius', type=int, default=20, help='Pocket extraction radius (default: 20)')
    args = parser.parse_args()

    split_pocket_ligand(args.input, args.protonate, args.radius)

