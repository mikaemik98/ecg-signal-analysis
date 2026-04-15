# ECG Arrhythmia Detector

## Overview

A machine learning model that classifies ECG heartbeats as normal or
abnormal using the MIT-BIH Arrhythmia Database. Achieves 98.7% accuracy
on 20,451 test beats.

## Dataset

- Source: MIT-BIH Arrhythmia Database (PhysioNet)
- 48 ECG recordings, each approximately 30 minutes long
- 360 Hz sampling frequency
- 102,251 individual beats extracted and labeled
- Two classes: Normal (75,028) and Abnormal (27,223)

## Results

| Metric           | Normal | Abnormal |
| ---------------- | ------ | -------- |
| Precision        | 0.99   | 0.99     |
| Recall           | 1.00   | 0.96     |
| F1 Score         | 0.99   | 0.98     |
| Overall Accuracy | 98.7%  |          |

## Confusion Matrix

|                   | Predicted Normal | Predicted Abnormal |
| ----------------- | ---------------- | ------------------ |
| Actually Normal   | 14,971 ✅        | 35 ⚠️              |
| Actually Abnormal | 224 ❌           | 5,221 ✅           |

- 14,971 normal beats correctly cleared
- 5,221 abnormal beats correctly flagged
- 35 false alarms on normal beats (0.2%)
- 224 abnormal beats missed (4%)

## Signal Processing Pipeline

1. Load raw ECG signal from PhysioNet format
2. Clean signal using bandpass filter (removes noise and baseline wander)
3. Detect R peaks using neurokit2
4. Calculate RR intervals and HRV metrics
5. Segment signal into 200 sample windows around each R peak
6. Classify each segment as normal or abnormal

## Key Findings

- Raw ECG signals contain significant noise from movement and breathing
  that must be filtered before analysis
- Normal and abnormal beats have visually distinct shapes especially
  in the P wave before the R peak
- Random Forest on raw signal segments achieves 98.7% accuracy
- Model makes the right clinical errors — very few false alarms (0.2%)
  with acceptable miss rate (4%)
- HRV analysis revealed average SDNN of 51.4ms indicating healthy
  autonomic nervous system function in record 100

## What I Learned

- ECG signal structure — PQRST complex, R peaks, HRV
- Signal filtering — removing noise while preserving heartbeat information
- Beat segmentation — cutting continuous signals into individual windows
- HRV metrics — SDNN and RR interval analysis
- Working with PhysioNet medical signal format using wfdb
- How arrhythmias appear differently in signal shape vs timing
- Precision, recall and F1 score for imbalanced medical classification

## Connection to CardioRest

The HRV metrics calculated here (RR intervals, SDNN) are the same
metrics used in the CardioRest HRV sleep quality monitoring application,
showing the clinical relevance of this analysis pipeline.

## How to Run

1. Create PhysioNet account and download MIT-BIH Arrhythmia Database
   https://physionet.org/content/mitdb/1.0.0/
2. Extract to data/mit-bih-arrhythmia-database-1.0.0/
3. Install dependencies:
   pip install wfdb neurokit2 scipy matplotlib pandas numpy scikit-learn
4. Run exploration.ipynb for signal visualization and analysis
5. Run model.ipynb for beat classification

## Project Structure

ecg-signal-analysis/
├── data/ ← download dataset here (not included)
├── exploration.ipynb
├── model.ipynb
└── README.md

## Tools Used

Python, wfdb, neurokit2, SciPy, Scikit-learn, Matplotlib, Jupyter, GitHub
