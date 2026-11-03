"""
Featurizer for the web app: the same code as featurize() in
Part1_MILES-HUANG-AREFIN-KEYA-LIANANNIE-LIN-ANYA-WEERANARAWAT.py, copied verbatim
(4,273 features: ECFP4 2048, FCFP 1024, atom-pair 1024, 10 descriptors, MACCS 167).
test_app_parity.py checks that it returns identical features to the training script.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors, Lipinski, MACCSkeys

# ECFP4: 2048 bits
ecfp_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# FCFP: 1024 bits
fcfp_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024,atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen())

# Atom-pair: 1024 bits
atom_pair_generator = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=1024)

def molecular_descriptors(mol):
   return np.array([
       Descriptors.MolWt(mol),
       Descriptors.MolLogP(mol),
       Descriptors.TPSA(mol),
       Descriptors.MolMR(mol),
       Lipinski.NumHDonors(mol),
       Lipinski.NumHAcceptors(mol),
       Lipinski.NumRotatableBonds(mol),
       Lipinski.RingCount(mol),
       Descriptors.FractionCSP3(mol),
       Descriptors.HeavyAtomCount(mol),
   ], dtype=np.float32)

# def molecular_descriptors(mol):
#     values = []

#     for name, function in Descriptors._descList:
#         try:
#             value = function(mol)

#             if np.isfinite(value):
#                 values.append(value)
#             else:
#                 values.append(0.0)

#         except:
#             values.append(0.0)

#     return np.array(values, dtype=np.float32)


def featurize(smiles):
    """TODO 3: SMILES strings -> 2D float32 NumPy array,
    one row per molecule.
    """

    def fingerprint(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:  # unparseable SMILES -> all-zero vector instead of crashing
            return np.zeros(2048 + 1024 + 1024 + 10 + 167, dtype=np.float32)

        ecfp = ecfp_generator.GetFingerprintAsNumPy(mol).astype(np.float32)
        fcfp = fcfp_generator.GetFingerprintAsNumPy(mol).astype(np.float32)
        atom_pair = atom_pair_generator.GetFingerprintAsNumPy(mol).astype(np.float32)
        descriptors = molecular_descriptors(mol)
        maccs = np.array(list(MACCSkeys.GenMACCSKeys(mol)),dtype=np.float32)

        return np.hstack([ecfp, fcfp, atom_pair, descriptors, maccs])

    return np.stack([fingerprint(s) for s in smiles])
