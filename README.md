# Cross-Subject N400 Decoding from Single-Trial EEG

Undergraduate deep learning final project.

We investigate whether convolutional neural networks can decode
word-level predictability from single-trial EEG in a cross-subject
setting — training on one set of participants and evaluating on
individuals the model has never seen. The primary signal of interest
is the N400 effect: a centroparietal EEG negativity peaking ~400 ms
after a semantically unexpected word, whose amplitude varies
inversely with cloze probability.

---

## Questions Explored

| #   | Question                                                                                                                          | Category                 |
| --- | --------------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| Q1  | Can deep learning decode cloze predictability from single-trial EEG cross-subject?                                                | Feasibility              |
| Q2  | Which temporal epoch contains the most decodable signal?                                                                          | Temporal analysis        |
| Q3  | Does preserving temporal position in the classifier improve decoding over global temporal pooling?                                | Architectural comparison |
| Q4  | Does a domain-specific architecture (EEGNet) outperform a general CNN, and at what training set size does any advantage diminish? | Architecture benchmark   |
| Q5  | What is the minimum number of training subjects for above-chance, architecturally stable decoding?                                | Data efficiency          |
| Q6  | Do representations learned on 18 subjects generalise to completely held-out individuals?                                          | Generalisation           |

---

## Dataset

**DERCo** (Quach et al., 2024) — available on OpenNeuro.

- 20 participants (2 excluded by dataset authors for excessive ocular
  artefacts prior to release)
- Five short stories from Grimm's Fairy Tales, self-paced reading
- 32 EEG channels, originally at 1000 Hz
- 28,265 single-trial epochs after preprocessing
- Binary labels: **high-cloze** vs **low-cloze**

Cloze probability labels were collected from **500 separate
crowdsourced participants** in a behavioural word-prediction task,
independent of the EEG sample.

### Preprocessing

**Provided by DERCo authors:**

- FIR bandpass filtering
- Common average referencing (CAR)
- FASTER automated artefact rejection
- Independent component analysis (ICA)
- Autoreject (epoch-level artefact removal)

**What we did:**

- Downsampled 1000 Hz → 250 Hz (Kaiser window anti-aliasing)
- Epoch extraction at four theoretically motivated time windows:
  `0–800 ms`, `0–200 ms`, `300–500 ms`, `500–800 ms`
- Binary label creation (high- vs low-cloze threshold)
- 2 subjects reserved as held-out test set prior to any analysis
- Z-scoring per CV fold using training fold statistics only

---

## Architectures

Three architectures addressing complementary questions.

### CNN-v1 — Temporal-agnostic baseline (Q3)

Tests whether global amplitude summarised over time carries
decodable predictability information.

Input: (batch, 1, 32, T)
Temporal conv: kernel (1×31), 8 filters, bias=False
BatchNorm + ELU
AvgPool: (1×4), reduces T → T/4
Spatial conv: kernel (32×1), 16 filters, bias=False
BatchNorm + ELU
Global AvgPool: (1×1) → 16 scalars
Classifier: Linear(16→32) → ELU → Linear(32→1)

### CNN-v2 — Temporal-sensitive model (Q3)

Tests whether preserving temporal position in the classifier improves
on global summarisation. Three simultaneous changes from CNN-v1:
doubled filter depth, stacked temporal convolutions, global average
pooling replaced with flatten.

Input: (batch, 1, 32, T)
Temporal conv 1: kernel (1×31), 16 filters, bias=False
BatchNorm + ELU
Temporal conv 2: kernel (1×15), 16 filters, bias=False
BatchNorm + ELU
AvgPool: (1×4), reduces T → T/4
Spatial conv: kernel (32×1), 32 filters, bias=False
BatchNorm + ELU
Flatten: → 1,600 values (for T=200)
Classifier: Linear(1600→32) → ELU → Linear(32→1)

### EEGNet — Domain-specific architecture (Q4)

Adapted from Lawhern et al. (2018). Core architecture unchanged.
Two modifications: output head changed from N-class to single logit
(BCEWithLogitsLoss for binary classification); classifier size
computed dynamically via dummy forward pass to support variable
epoch lengths.

Input: (batch, 1, 32, T)
Temporal conv: kernel (1×125), 8 filters
Depthwise spatial: kernel (32×1), groups=F1
AvgPool ÷4
Separable conv: kernel (1×31)
AvgPool ÷8
Flatten → Linear(→1)

## Training Protocol

**Evaluation:** 5-fold subject-disjoint cross-validation on 18
train/val subjects. All trials from test-fold subjects are held out
entirely per fold (~3–4 subjects per test fold). 2 subjects reserved
as a completely held-out out-of-sample test set.

| Parameter      | Value                                      |
| -------------- | ------------------------------------------ |
| Loss           | BCEWithLogitsLoss                          |
| Optimiser      | AdamW (lr=1×10⁻³, wd=1×10⁻⁴)               |
| Scheduler      | ReduceLROnPlateau                          |
| Early stopping | Patience 5 on validation AUC               |
| Normalisation  | Z-score per fold, training statistics only |
| Max epochs     | 100 (typically converges in 5–20)          |
| Primary metric | Out-of-fold ROC AUC                        |

## References

Quach, B. et al. (2024). DERCo: A Dataset for Human
Behaviour-Centred Explainability in Reading Comprehension with
Eye-Tracking and EEG. _arXiv:2309.02079._

Lawhern, V. J. et al. (2018). EEGNet: A compact convolutional neural
network for EEG-based brain-computer interfaces. _Journal of Neural
Engineering, 15_(5), 056013.

Kutas, M. & Hillyard, S. A. (1980). Reading senseless sentences:
Brain potentials reflect semantic incongruity. _Science, 207_(4427),
203–205.

Gramfort, A. et al. (2014). MNE software for processing MEG and EEG
data. _NeuroImage, 86_, 446–460.
