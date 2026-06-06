# Cross-Subject EEG Decoding of Word Predictability

Undergraduate deep learning project for class.

## Overview

This project investigates whether a simpler CNN can decode word predictability from single-trial EEG in a subject-disjoint evaluation. The question being asked is, "Can this CNN generalize to participants it has never seen during training?"

## Questions Driving the Exploration

1. Can a CNN decode high vs low cloze probability from single-trial EEG across unseen participants?
2. Which temporal window carries the most decodable (to the CNN) information about word predictability?
3. Does a domain-specific architecture (EEGNet) outperform a simpler CNN on the full 0-800 ms window?

## Dataset

DERCo: A dataset for human behavior in reading EEG experiments with crowdsourcing of next-word predictions.

20 participants (2 excluded for eye movement artifacts), 5 articles, 32 EEG channels, 1000 Hz. Downsampled to 250 Hz to explore question #3.

## Architecture

- Custom two-stage CNN: (1) temporal convolution (125 ms kernel) and then (2) spatial convlution across all 32 channels
- EEGnet (Lawhern et al., 2018) for architectural comparison on 0-800 ms window.

## Evaluation

5-fold subject-disjoint cross-validation on 18 train/val participants.
2 participants for out-sample testing.
Primary metric used: ROC AUC due to balanced classes. Secondary metric used: balanced accuracy.

## References

Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung,
C. P., & Lance, B. J. (2018). EEGNet: A compact convolutional neural
network for EEG-based brain-computer interfaces. Journal of Neural
Engineering, 15(5), 056013.

Quach, B. M., Gurrin, C., & Healy, G. (2024). DERCo: A dataset for
human behaviour in reading comprehension using EEG. Scientific Data,
11(1), 1104. Nature Publishing Group.
