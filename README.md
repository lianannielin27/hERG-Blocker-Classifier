# 203_HW1 — hERG blocker classifier

**Web app:** https://herg-blocker-predictor.onrender.com/ (hosted on Render)

**Group members:** [Miles Huang, Annie Li An Lin, Arefin Keya, Anya Weeranarawat]

## Decision threshold

We classify a molecule as a hERG blocker when the model's probability is at least **0.45**. The errors are asymmetric: a false negative lets a cardiotoxic compound advance through development at great cost and possible harm to patients, while a false positive only triggers a cheap confirmatory hERG assay. So we accept more false positives and set the threshold below 0.5. On the validation set (1,000 molecules, model fit on the training set only) we compared candidate rules: the lowest threshold reaching sensitivity ≥ 0.90 (0.43, specificity 0.75), Youden's J (0.51, sensitivity 0.86, specificity 0.84), and maximum sensitivity with specificity above 0.7 (0.39, specificity 0.71). 0.45 sits between the first two: on validation it gives sensitivity 0.89, specificity 0.79 and precision 0.82, and on our held-out test split sensitivity 0.89, specificity 0.80, precision 0.83, accuracy 0.85 (ROC-AUC 0.919). Scaffold-grouped cross-validation, which better imitates a test set of different composition, gave lower numbers at 0.45 (sensitivity 0.84, specificity 0.70, precision 0.75, accuracy 0.77), so we expect real performance nearer those, and we did not go lower than 0.45 because specificity would approach the 0.7 floor.

## Features

Each molecule's SMILES is parsed with RDKit and turned into a fixed-length vector of 4,273 values, so SMILES of any length work. The vector concatenates 2,048-bit Morgan ECFP4 fingerprints (radius 2, local substructures and scaffold diversity), 1,024-bit FCFP fingerprints (Morgan with atoms abstracted to functional roles such as donor, acceptor, aromatic and charged, so similar amines look alike across chemotypes), 1,024-bit atom-pair fingerprints (pairs of atoms and the bond distance between them, which capture overall shape and spacing), ten RDKit descriptors (molecular weight, logP, TPSA, molar refractivity, hydrogen-bond donors and acceptors, rotatable bonds, ring count, fraction of sp3 carbons and heavy-atom count, covering the lipophilicity, bulk and flexibility associated with hERG binding), and the 167 MACCS structural keys. We did not run ChemBERTa embeddings.

## Model and hyperparameters

The model is a scikit-learn **RandomForestClassifier** with 450 trees, scikit-learn defaults for everything else (fully grown trees, `max_features="sqrt"`, `min_samples_leaf=1`) and seed 42, trained on the training and validation molecules together and exported to ONNX (47.7 MB). We reduced the forest from 500 to 450 trees because 500 fully grown trees exported to 53.1 MB, over the 50 MB limit, and the test-split ROC-AUC stayed at 0.919. Five-fold cross-validated AUROC on the 8,000 development molecules is 0.916 ± 0.004 with random folds and 0.855 ± 0.016 with scaffold-grouped folds (no scaffold on both sides of a split), so about 0.86 is the more realistic figure for molecules unlike the training set. We chose a random forest over other models using an earlier comparison on a similar fingerprint-and-descriptor feature set (not exactly these features), where random forest, ExtraTrees and HistGradientBoosting tied (AUROC 0.905 to 0.910 in 5-fold CV) and logistic regression was clearly worse (0.853); HistGradientBoosting could not be exported with the allowed skl2onnx version, which left the random forest.

## Use of LLMs

We used Claude Code (Anthropic) to help with this project: 
