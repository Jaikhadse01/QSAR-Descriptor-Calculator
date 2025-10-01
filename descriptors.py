# descriptors.py
from rdkit import Chem
from rdkit.Chem import Descriptors

def calculate_descriptors(smiles):
    """Return a dict of basic descriptors or None if invalid SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "Molecular Weight": round(Descriptors.MolWt(mol), 2),
        "LogP": round(Descriptors.MolLogP(mol), 2),
        "TPSA": round(Descriptors.TPSA(mol), 2),
        "H-bond Donors": Descriptors.NumHDonors(mol),
        "H-bond Acceptors": Descriptors.NumHAcceptors(mol),
        "Rotatable Bonds": Descriptors.NumRotatableBonds(mol)
    }
