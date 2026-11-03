"""
Homework 1, Part 1 - hERG blockade classifier.

Rename to Part1_[GROUP_NAMES].py, fill in the four TODOs, 
then run: python Part1_[GROUP_NAMES].py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors, Lipinski, MACCSkeys
from skl2onnx import to_onnx
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

# TODO 1: your group's names -> Part1_FIRST1-LAST1-FIRST2-LAST2.onnx
NAME = "MILES-HUANG-AREFIN-KEYA-LIANANNIE-LIN-ANYA-WEERANARAWAT"

# TODO 2: your own threshold; justify it in README.md
THRESHOLD = 0.45

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

        ecfp = ecfp_generator.GetFingerprintAsNumPy(mol).astype(np.float32)
        fcfp = fcfp_generator.GetFingerprintAsNumPy(mol).astype(np.float32)
        atom_pair = atom_pair_generator.GetFingerprintAsNumPy(mol).astype(np.float32)
        descriptors = molecular_descriptors(mol)
        maccs = np.array(list(MACCSkeys.GenMACCSKeys(mol)),dtype=np.float32)

        return np.hstack([ecfp, fcfp, atom_pair, descriptors, maccs])

    return np.stack([fingerprint(s) for s in smiles])

def build_model():
    """TODO 4: Return your model object here.

    If you are using sklearn, then an unfitted sklearn classifier or Pipeline,
    e.g. RandomForestClassifier(...).

    Not using sklearn? Export your own ONNX with one input of shape
    (n_molecules, n_features) where n_molecules can vary (PyTorch: dynamic_axes),
    and a last output holding P(blocker), with sigmoid/softmax inside the model.
    featurize must take SMILES strings and return one row per molecule.
    TestModel.py checks this.
    """

    return RandomForestClassifier(n_estimators=450, random_state=42)


def save_onnx(model, sample):
    """
    Export a fitted model. `sample` is one row of your features,
    which fixes the input type.
    """
    Path("Part1_%s.onnx" % NAME).write_bytes(
        to_onnx(model, sample, options={"zipmap": False}, target_opset=12).SerializeToString())
    print("\nwrote Part1_%s.onnx" % NAME)


def load_split():
    """
    Load in assignment data and split into train/val/test sets
    You do not need to use this exact split, as you will be evaluated
    on a held-out test set. You may use the data in any way you wish.
    """
    df = pd.read_csv("data/hw1data.tab", sep="\t")
    df.columns = ["Drug_ID", "Drug", "Y"]
    test = df.sample(frac=0.2, random_state=42)
    rest = df.drop(test.index)
    valid = rest.sample(frac=0.125, random_state=1)
    return {"train": rest.drop(valid.index), "valid": valid, "test": test}


def prepare(frame):
    """Hand SMILES strings and labels to your featurizer."""
    return np.asarray(featurize(frame["Drug"].tolist()), np.float32), frame["Y"].to_numpy()


def report(y, probability, threshold):
    pred = (probability >= threshold).astype(int)
    tn, fp, _, _ = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    print("\nthreshold %.2f" % threshold)
    for name, value in [("accuracy", accuracy_score(y, pred)),
                        ("sensitivity/recall", recall_score(y, pred)),
                        ("specificity", tn / (tn + fp)),
                        ("precision", precision_score(y, pred)),
                        ("f1", f1_score(y, pred)),
                        ("ROC-AUC", roc_auc_score(y, probability)),
                        ("PR-AUC", average_precision_score(y, probability))]:
        print("  %-20s%.3f" % (name, value))


# def main():
#     split = load_split()

#     # Combine all of the data you currently have
#     df = pd.concat([
#         split["train"],
#         split["valid"],
#         split["test"]
#     ]).reset_index(drop=True)

#     X, y = prepare(df)

#     cv = StratifiedKFold(
#         n_splits=5,
#         shuffle=True,
#         random_state=42
#     )

#     train_aucs = []
#     valid_aucs = []

#     train_praucs = []
#     valid_praucs = []

#     for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):

#         X_train = X[train_idx]
#         y_train = y[train_idx]

#         X_valid = X[valid_idx]
#         y_valid = y[valid_idx]

#         model = build_model()
#         model.fit(X_train, y_train)

#         # Training probabilities
#         train_probability = model.predict_proba(X_train)[:, 1]

#         # Validation probabilities
#         valid_probability = model.predict_proba(X_valid)[:, 1]

#         train_auc = roc_auc_score(y_train, train_probability)
#         valid_auc = roc_auc_score(y_valid, valid_probability)

#         train_prauc = average_precision_score(
#             y_train,
#             train_probability
#         )

#         valid_prauc = average_precision_score(
#             y_valid,
#             valid_probability
#         )

#         train_aucs.append(train_auc)
#         valid_aucs.append(valid_auc)

#         train_praucs.append(train_prauc)
#         valid_praucs.append(valid_prauc)

#         print(f"\nFOLD {fold}")
#         print(f"Train ROC-AUC:      {train_auc:.3f}")
#         print(f"Validation ROC-AUC: {valid_auc:.3f}")
#         print(f"Train PR-AUC:       {train_prauc:.3f}")
#         print(f"Validation PR-AUC:  {valid_prauc:.3f}")

#     print("\n==============================")
#     print("5-FOLD CROSS-VALIDATION")
#     print("==============================")

#     print(
#         "Mean validation ROC-AUC:",
#         f"{np.mean(valid_aucs):.3f} +/- {np.std(valid_aucs):.3f}"
#     )

#     print(
#         "Mean validation PR-AUC:",
#         f"{np.mean(valid_praucs):.3f} +/- {np.std(valid_praucs):.3f}"
#     )

#     print(
#         "Mean training ROC-AUC:",
#         f"{np.mean(train_aucs):.3f}"
#     )

#     print(
#         "Mean training PR-AUC:",
#         f"{np.mean(train_praucs):.3f}"
#     )

def main():
    """
    You do not have to use the train/test split provided to you.
    This is just an example of splitting the data.
    You may use the entire hw1data.tab as you wish.
    The dataset used to evaluate your model is not provided to you (just
    like in the real world - you train on some data then deploy your model,
    which will receive new unseen data).
    """
    split = load_split()

    X_train, y_train = prepare(
        pd.concat([split["train"], split["valid"]])
    )
    X_test, y_test = prepare(split["test"])

    model = build_model().fit(X_train, y_train)

    probability = model.predict_proba(X_test)[:, 1]

    report(y_test, probability, 0.5)
    report(y_test, probability, THRESHOLD)

    save_onnx(model, X_test[:1])


if __name__ == "__main__":
    main()

