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

We selected a Random Forest classifier with 500 decision trees (n_estimators=500, random_state=42), using default settings for maximum tree depth, minimum samples per leaf, and feature selection at each split. ROC-AUC was our primary metric for model selection because it measures the model's overall ability to distinguish hERG blockers from non-blockers across classification thresholds. Given the drug safety-screening objective, we also considered sensitivity and specificity to minimize missed blockers while limiting false positives. Our final model achieved a validation ROC-AUC of approximately 0.927, with five-fold cross-validation yielding a mean ROC-AUC of 0.925 ± 0.004. On the held-out test set, it achieved a ROC-AUC of 0.919, sensitivity of 0.891, and specificity of 0.800 at our selected threshold of 0.45. 

## Limitation

Although the larger feature set and 500-tree ensemble may improve predictive performance and stability, they also increase computational cost and model complexity. The unrestricted tree depth and minimum leaf size of one may allow individual trees to overfit training data, particularly with many potentially redundant features. Random Forest averaging helps reduce this risk, but does not eliminate it.

## Deployment

Render’s free tier allows 512 MB of RAM, and onnxruntime alone used about 415 MB to load the forest, leaving too little room for Gradio and RDKit. We replaced onnxruntime with a small NumPy tree evaluator (light_onnx.py in Part 2) that reads the same .onnx file.
The NumPy evaluator gave the same results as onnxruntime: probabilities differed by at most 2.4e-6 across 4,004 molecules, and no labels changed at the 0.45 threshold.

## Use of LLMs

We used Claude Code and ChatGPT Enterprise to help with code revision, debugging, and deploying troubleshoot.  Claude code helped build the Gradio app, write tests to compare and fix render deployment. We made modeling and feature decisions ourselves while using AI to keep us better informed, such as comparing threshold options and running checks. AI generated code has been reviewed prior to submission.
