from rdkit import Chem
import torch
import os
from tqdm import tqdm

def sdf_to_tensor(sdf_path):
    # Read the molecule from the SDF file
    suppl = Chem.SDMolSupplier(sdf_path)
    mol = [mol for mol in suppl][0]

    # Extract atomic coordinates
    coords = mol.GetConformer().GetPositions()

    # Convert coordinates to a tensor
    tensor = torch.tensor(coords)

    return tensor

# Define directory where all your SDF files are stored
sdf_directory = "sdf"

# List of all SDF files
sdf_files = [file for file in os.listdir(sdf_directory) if file.endswith(".sdf")]

# Iterate over each SDF file and save the tensor
for file in tqdm(sdf_files, desc="Converting SDF to tensors"):
    tensor = sdf_to_tensor(os.path.join(sdf_directory, file))
    tensor_filename = os.path.splitext(file)[0] + '.pt' # Replacing .sdf with .pt (PyTorch tensor extension)
    torch.save(tensor, os.path.join(sdf_directory, tensor_filename))
