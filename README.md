# Neuromorphic Emotion Recognition from EEG

Predicting **valence** and **arousal** from EEG with a spiking neural network (SNN),
using the [DEAP dataset](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/).
The SNN is benchmarked against an SVM and Logistic Regression on identical features and folds.

## Overview
- **Data:** DEAP, 32 participants x 40 one-minute music videos, 32 EEG channels at 128 Hz.
- **Labels:** valence and arousal ratings (1-9) split at 5 into low/high.
  Combined, they map to four emotion quadrants (happy/excited, calm/relaxed, angry/stressed, sad/bored).
- **Features:** log band power (theta, alpha, beta, gamma) per channel, giving 128 features per 4-second window.
- **Model:** 128 inputs -> 256 LIF neurons -> 2 LIF output neurons, run for 15 time steps,
  trained with surrogate gradients (snnTorch). The prediction is the output neuron with more spikes.

## Pipeline
1. Keep the 32 EEG channels, remove the 3 s baseline
2. Bandpass filter 4-45 Hz, normalise per channel
3. Cut 4 s windows with 50% overlap (1,160 windows per subject, 37,120 in total)
4. Extract band-power features, standardise per subject
5. Train and evaluate SVM, Logistic Regression and SNN

Splits are always made by **trial**, so windows from one trial never appear in both train and test.

## Results

**Within-subject** (train and test on the same person, 32 subjects, 4-fold split by trial, balanced accuracy; chance = 0.50):

| Model | Valence | Arousal |
|---|---|---|
| SVM (RBF) | 0.596 | 0.536 |
| Logistic Regression | 0.584 | 0.543 |
| **SNN** | **0.599** | 0.538 |

**Cross-subject** (train on subjects 1-24, test on unseen subjects 25-32):

| Model | Valence | Arousal |
|---|---|---|
| Logistic Regression (accuracy) | 0.527 | 0.562 |
| SNN (balanced accuracy, last 3 epochs) | 0.518 | 0.522 |

**Sparsity:** about 16% of hidden neurons spike per time step within-subject, and 3-4% cross-subject.

**Four-emotion output** (valence and arousal models combined, unseen subjects): 0.29 accuracy, shown as a pipeline demo.

### Takeaways
- The SNN matches an SVM and Logistic Regression; the differences are within noise.
- Valence is decodable within a person (about 60%, up to 0.76 for some individuals). Arousal is weaker.
- Accuracy does not transfer to unseen people (about 52%), which is typical for EEG.

### Limitations
- Only 40 trials per person, and no separate validation set.
- Features were standardised per subject before the folds were split. This affects all models equally.

## Project structure
```
.
├── preprocessed.ipynb   # preprocessing, band-power features, SVM baseline
├── notebook.ipynb       # features, SNN training, within/cross-subject evaluation
├── load_data.py         # loads a DEAP subject file
├── preprocess.py        # filtering, windowing, band-power features
├── snn_model.py         # SNN model (snnTorch)
└── data/                # not included (see below)
```

## Setup
```bash
pip install numpy scipy scikit-learn matplotlib torch snntorch kagglehub
```

### Data
The dataset is not included. Download it from Kaggle:
```python
import kagglehub
path = kagglehub.dataset_download("guneethpalakurthy/deap-dataset")
```
Copy `s01.dat` ... `s32.dat` into a `data/` folder. Use of DEAP is subject to its own license.

### Run
1. Run `preprocessed.ipynb` to create `data/processed/s01.npz ... s32.npz`.
2. Run `notebook.ipynb` to build `features_all.npz`, train the SNN and reproduce the results.

## Future work
Differential-entropy features, leave-one-subject-out evaluation, recurrent or convolutional SNNs,
a web demo, and deployment on neuromorphic hardware.

## References
- Koelstra et al., "DEAP: A Database for Emotion Analysis using Physiological Signals," IEEE Trans. Affective Computing, 2012.
- Eshraghian et al., "Training Spiking Neural Networks Using Lessons From Deep Learning" (snnTorch).
- Russell, "A Circumplex Model of Affect," 1980.
