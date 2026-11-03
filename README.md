# 203_HW1 — hERG blocker classifier

**Web app:** https://herg-blocker-predictor.onrender.com/ (hosted on Render)

**Group members:** [Miles Huang, Li An Annie Lin, Arefin Keya, Anya Weeranarawat]

## Introduction

We developed a machine-learning classifier to predict whether a molecule is an hERG blocker from its SMILES string. Because hERG blockade can contribute to cardiotoxicity risk during drug development, identifying potential blockers is an important part of drug safety screening. Our project integrates molecular feature engineering, model selection and tuning, and deployment into a single pipeline. We convert SMILES strings into numerical features, use those features to train a random forest classifier, and make the model accessible through a web app where users can enter a SMILES string and receive a prediction.

## Decision threshold

We chose a decision threshold of 0.45 to balance sensitivity and specificity while prioritizing the detection of hERG blockers. Lowering the threshold below 0.5 flags more molecules as blockers, reducing false negatives at the cost of more false positives. At 0.45, our validation sensitivity was 0.89 and specificity was 0.79. Further decreasing the threshold yielded diminishing returns in sensitivity relative to the loss in specificity. On the test set, lowering the cutoff from 0.45 to 0.40 increased sensitivity from 0.891 to 0.910 while specificity fell from 0.800 to 0.752. We favored 0.45 because it retained high sensitivity without accepting this larger specificity loss for a modest sensitivity gain. ROC-AUC remained unchanged because it evaluates ranking across thresholds rather than performance at one chosen cutoff.

## Features

We seleceted features incrementally based on their ability to improve ROC-AUC and PR-AUC while trying to preserve the highest sensitivity, recall, precision, and accuracy. Each SMILES string is converted into a fixed-length numerical representation using several complmentary RDKIT features:
- 2,048-bit ECFP4 (Morgan) fingerprints: capture local atomic environements around each atoms out to a radius of two bonds and encodes those substructures into a fixed-length fingerprint 
- 1,024-bit FCFP fingerprint: represents atoms by functional chemical properties to provide a complementary view of the moelcule's functional group in each atom's local neighborhood
- 1,024 atom-pair fingerprint: consideres the topological distances between each atom-pair, providing information on chemical feature arrangement
- molecular descriptors: a summary of 10 global physicochemical properties including molecular size, lipophilicity, etc.
- MACCs strucutral keys: encodes whether a predefined set of chemical structural patterns are present or absent

We began with ECFP alone because Morgan fingerprints were introduced in lecture, then tested the other feature groups incrementally and retained features when they improved validation performance. The different representations were intended to provide complementary information: ECFP and FCFP describe local molecular environments, atom-pair fingerprints describe relationships across the molecular graph, MACCS keys identify known structural patterns, and molecular descriptors summarize whole-molecule properties. The combined feature set improved validation ROC-AUC from approximately 0.906 with ECFP alone to 0.927 and PR-AUC from 0.914 to 0.936. Five-fold stratified cross-validation produced a mean ROC-AUC of 0.925 ± 0.004 and mean PR-AUC of 0.933 ± 0.003, suggesting that the improvement was consistent across different data splits.


## Model and hyperparameters

The model is a scikit-learn **RandomForestClassifier** with 450 trees, scikit-learn defaults for everything else (fully grown trees, `max_features="sqrt"`, `min_samples_leaf=1`) and seed 42, trained on the training and validation molecules together and exported to ONNX (47.7 MB). We reduced the forest from 500 to 450 trees because 500 fully grown trees exported to 53.1 MB, over the 50 MB limit, and the test-split ROC-AUC stayed at 0.919. Five-fold cross-validated AUROC on the 8,000 development molecules is 0.916 ± 0.004 with random folds and 0.855 ± 0.016 with scaffold-grouped folds (no scaffold on both sides of a split), so about 0.86 is the more realistic figure for molecules unlike the training set. We chose a random forest over other models using an earlier comparison on a similar fingerprint-and-descriptor feature set (not exactly these features), where random forest, ExtraTrees and HistGradientBoosting tied (AUROC 0.905 to 0.910 in 5-fold CV) and logistic regression was clearly worse (0.853); HistGradientBoosting could not be exported with the allowed skl2onnx version, which left the random forest.

## Use of LLMs

[CONFIRM AND EDIT] We used Claude Code (Anthropic) to help with this project: it built the Gradio web app and the app-versus-ONNX parity test, ran the cross-validation, scaffold-split and threshold analyses quoted above, diagnosed and fixed the ONNX size problem (500 to 450 trees), and drafted this README. [State here what the team did yourselves: who wrote the featurization and model code, what you reviewed, which experiments you reran, and that you checked every number in this README against the logs.]
