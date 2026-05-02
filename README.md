# TTC Delay Prediction

A machine learning project analyzing Toronto Transit Commission (TTC) 
delay data to predict delay risk and duration using classification 
and regression models.

## Project Overview

Using the TTC 2025 delay dataset, we built models to:
- **Classify** whether a delay is high-risk (delay > 30 minutes = 1, otherwise = 0)
- **Predict** the actual delay duration in minutes using regression

## Models & Results

### Classification
| Model | Accuracy |
|-------|----------|
| Linear Discriminant Analysis (LDA) | 68% |
| Quadratic Discriminant Analysis (QDA) | 67% |
| Logistic Regression | 68% |
| Decision Tree | 78% |
| K-Nearest Neighbour (KNN) | 70% |

### Regression
| Model | MSE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | 83.66 | 9.15 | 0.03 |
| Decision Tree | 61.16 | 7.82 | 0.29 |
| Random Forest | 50.68 | 7.12 | 0.41 |

## Authors
- Arbert
- Jeryshan

*Developed as part of a course project at Wilfrid Laurier University.*


