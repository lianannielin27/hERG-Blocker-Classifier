# 203_HW1 — hERG blocker classifier

**Web app:** https://herg-blocker-predictor.onrender.com/ (hosted on Render)

**Group members:** [Miles Huang, Annie Li An Lin, Arefin Keya, Anya Weeranarawat]

## Introduction

We developed a machine-learning classifier to predict whether a molecule is an hERG blocker from its SMILES string. Because hERG blockade can contribute to cardiotoxicity risk during drug development, identifying potential blockers is an important part of drug safety screening. Our project integrates molecular feature engineering, model selection and tuning, and deployment into a single pipeline. We convert SMILES strings into numerical features, use those features to train a random forest classifier, and make the model accessible through a web app where users can enter a SMILES string and receive a prediction.

## Decision threshold

We chose a decision threshold of 0.45 to balance sensitivity and specificity while prioritizing the detection of hERG blockers. Lowering the threshold below 0.5 flags more molecules as blockers, reducing false negatives at the cost of more false positives. At 0.45, our validation sensitivity was 0.89 and specificity was 0.79. Further decreasing the threshold yielded diminishing returns in sensitivity relative to the loss in specificity. On the test set, lowering the cutoff from 0.45 to 0.40 increased sensitivity from 0.891 to 0.910 while specificity fell from 0.800 to 0.752. We favored 0.45 because it retained high sensitivity without accepting this larger specificity loss for a modest sensitivity gain. ROC-AUC remained unchanged because it evaluates ranking across thresholds rather than performance at one chosen cutoff.

## Features

Each molecule's SMILES is parsed with RDKit and turned into a fixed-length vector of 4,273 values, so SMILES of any length work. The vector concatenates 2,048-bit Morgan ECFP4 fingerprints (radius 2, local substructures and scaffold diversity), 1,024-bit FCFP fingerprints (Morgan with atoms abstracted to functional roles such as donor, acceptor, aromatic and charged, so similar amines look alike across chemotypes), 1,024-bit atom-pair fingerprints (pairs of atoms and the bond distance between them, which capture overall shape and spacing), ten RDKit descriptors (molecular weight, logP, TPSA, molar refractivity, hydrogen-bond donors and acceptors, rotatable bonds, ring count, fraction of sp3 carbons and heavy-atom count, covering the lipophilicity, bulk and flexibility associated with hERG binding), and the 167 MACCS structural keys. We did not run ChemBERTa embeddings.

## Model and hyperparameters

The model is a scikit-learn **RandomForestClassifier** with 450 trees, scikit-learn defaults for everything else (fully grown trees, `max_features="sqrt"`, `min_samples_leaf=1`) and seed 42, trained on the training and validation molecules together and exported to ONNX (47.7 MB). We reduced the forest from 500 to 450 trees because 500 fully grown trees exported to 53.1 MB, over the 50 MB limit, and the test-split ROC-AUC stayed at 0.919. Five-fold cross-validated AUROC on the 8,000 development molecules is 0.916 ± 0.004 with random folds and 0.855 ± 0.016 with scaffold-grouped folds (no scaffold on both sides of a split), so about 0.86 is the more realistic figure for molecules unlike the training set. We chose a random forest over other models using an earlier comparison on a similar fingerprint-and-descriptor feature set (not exactly these features), where random forest, ExtraTrees and HistGradientBoosting tied (AUROC 0.905 to 0.910 in 5-fold CV) and logistic regression was clearly worse (0.853); HistGradientBoosting could not be exported with the allowed skl2onnx version, which left the random forest.

## Use of LLMs

[CONFIRM AND EDIT] We used Claude Code (Anthropic) to help with this project: it built the Gradio web app and the app-versus-ONNX parity test, ran the cross-validation, scaffold-split and threshold analyses quoted above, diagnosed and fixed the ONNX size problem (500 to 450 trees), and drafted this README. [State here what the team did yourselves: who wrote the featurization and model code, what you reviewed, which experiments you reran, and that you checked every number in this README against the logs.]
