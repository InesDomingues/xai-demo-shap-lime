# Similar Cases + Decision Tree Demo

This is a simple Streamlit web application designed to demonstrate interpretable machine learning using the Breast Cancer Wisconsin dataset.

The app trains a Decision Tree classifier and explains its predictions using two complementary approaches:

1. a visible decision tree structure;
2. a similar-cases explanation based on nearest neighbours.

## Purpose

This application was created for educational purposes.

It allows students to explore how a machine learning model can make predictions and how those predictions can be interpreted using both rule-based and example-based explanations.

The main pedagogical idea is to compare two questions:

- **What rule path did the decision tree follow?**
- **Which previous cases are most similar to the selected case?**

## Dataset

The application uses the Breast Cancer Wisconsin dataset available through `scikit-learn`.

The dataset contains features computed from digitised images of fine needle aspirates of breast masses.

The classification task is to distinguish between malignant and benign tumours.

Target classes:

- `0 = malignant`
- `1 = benign`

## What the app does

The app allows users to:

- inspect the dataset;
- train a Decision Tree classifier;
- adjust the complexity of the tree;
- visualise the decision tree;
- select an individual test case;
- see the true class and predicted class;
- inspect predicted probabilities;
- view the selected case values;
- find the most similar training cases;
- inspect the class distribution among similar cases;
- visualise the selected case and similar cases using PCA;
- inspect feature importance from the decision tree.

## Decision Tree

A Decision Tree classifier is used because it is easier to interpret than many other machine learning models.

Each prediction follows a sequence of decision rules. These rules are displayed in the tree diagram.

In the tree visualisation:

- malignant nodes are shown in red;
- benign nodes are shown in green.

This colour scheme is used to make the visualisation more intuitive for teaching.

## Similar Cases Explanation

The app also provides an example-based explanation.

For a selected test case, the app searches the training dataset for the most similar cases.

Similarity is calculated using Euclidean distance after standardising all features.

Standardisation is important because the features have different numerical scales. Without standardisation, variables with larger values could dominate the distance calculation.

## PCA Visualisation

The Breast Cancer Wisconsin dataset has many features.

For this reason, PCA is used to project the data into two dimensions for visualisation.

Important note:

PCA is used only for visualisation.

The similarity calculation is performed using all standardised features.

## Important Caution

This app is intended for educational purposes only.

The decision tree and the similar-cases explanation describe the behaviour of the model and the structure of the dataset.

They should not be interpreted as direct clinical evidence.

They do not prove clinical causality.

Similar cases are mathematically similar according to the chosen distance metric, but this does not necessarily mean they are clinically equivalent.

## How to run locally

Install the required packages:

```bash
pip install -r requirements.txt
