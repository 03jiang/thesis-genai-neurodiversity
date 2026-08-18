# Analysis

This directory contains the archived analysis code associated with Chapter 4 of the final submitted thesis.

The final submitted thesis PDF is the authoritative source for reported results. Earlier README files and Chapter 4 drafts contained obsolete intermediate analyses; conflicting labels, statistics, figures, and interpretations from those files are not preserved here.

The scripts in `scripts/` implement preprocessing and feature encoding, the 18 binary variables derived from Questions 1--4, K-Means and K-Modes clustering, candidate values of *k* from 2 to 10, silhouette and elbow/cost diagnostics, Adjusted Rand Index at *k* = 6, the Question 5 contingency analysis, Monte Carlo chi-square procedure, Cramér's V, and PCA visualisation. Original seeds, parameters, encoding, sample selection, and statistical procedures have been retained.

Participant-level input files and generated row-level intermediate CSV files are not publicly distributed. Consequently, the archived pipeline cannot be rerun from this public repository alone.

The PCA analysis script is provided, but the participant-level projection figure is not included because each plotted point corresponds to an individual respondent and point shape encodes the Question 5 self-identification response. PCA was used only as a descriptive visualisation and was not interpreted as evidence of latent psychological dimensions.

The scripts in `additional_checks/` are exploratory or sensitivity analyses not reported as primary results in the final thesis.

Exact historical package versions were not fully recorded; the listed environment reflects the reproducibility environment prepared for the archived thesis code.
