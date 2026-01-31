# Food Classification & Coffee Sales Forecasting

## Overview
This project addresses two distinct challenges in Machine Learning:
1. **Food Classification:** Using deep features from **ResNet50** to classify food into 8 categories.
2. **Coffee Sales Forecasting:** Predicting daily sales using **ESN**, **LSTM**, and **Bi-LSTM** architectures.

## Final Results

### Classification Task (8 Classes)
| Model | Test Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **SVM** | 83.70% | 0.8377 | 0.8370 | 0.8367 |
| **Random Forest** | 84.59% | 0.8461 | 0.8459 | 0.8455 |
| **MLP** | 77.79% | 0.7780 | 0.7779 | 0.7754 |

### Time Series Task (Coffee Sales)
| Model | RMSE | MAE | R² |
| :--- | :--- | :--- | :--- |
| **ESN** | 45.6842 | 36.3227 | 0.8165 |
| **LSTM** | 0.8355 | 0.6899 | 0.8314 |
| **Bi-LSTM** | 0.9399 | 0.7781 | 0.7866 |

## Methodology
**Feature Extraction:** Pre-trained **ResNet50** (Global Average Pooling).
**Calorie Estimation:** Hard-coded mapping of food classes to caloric values.
**Forecasting:** Daily resampled sales data with Min-Max normalization.

## Report
📄 See `Report.pdf` for the full technical write-up.
