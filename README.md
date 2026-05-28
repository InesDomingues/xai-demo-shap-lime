# SHAP and LIME Demo

This is a Streamlit web application designed to demonstrate explainable artificial intelligence (XAI) using the Breast Cancer Wisconsin dataset.

The app trains a Random Forest classifier and explains its predictions using two popular XAI methods:

- SHAP — SHapley Additive exPlanations
- LIME — Local Interpretable Model-agnostic Explanations

## Purpose

This application was created for educational purposes. It allows students to explore how machine learning predictions can be explained using global and local interpretability techniques.

The app demonstrates that explainability methods can help us understand which features contribute most to a model prediction, while also highlighting that these explanations describe the behaviour of the model and do not necessarily represent causal clinical mechanisms.

## Dataset

The application uses the Breast Cancer Wisconsin dataset available through `scikit-learn`.

The dataset contains features computed from digitised images of fine needle aspirates of breast masses. The classification task is to distinguish between malignant and benign tumours.

Target classes:

- 0 = malignant
- 1 = benign

## What the app does

The app allows users to:

- inspect the dataset;
- train a Random Forest classifier;
- view the model accuracy;
- select an individual test case;
- see the true class and predicted class;
- inspect predicted probabilities;
- view a SHAP global explanation;
- view a SHAP local explanation for an individual prediction;
- view a LIME local explanation for an individual prediction.

## SHAP

SHAP estimates how much each feature contributes to a model prediction.

It is based on Shapley values from cooperative game theory. In this app, SHAP is used to show:

- global feature importance;
- local contribution of features for a selected case.

## LIME

LIME explains an individual prediction by creating small perturbations around a selected case and fitting a simple interpretable model locally.

In this app, LIME is used to explain why the trained Random Forest model made a specific prediction for a selected case.

## Important note

SHAP and LIME explain the behaviour of the machine learning model.

They do not prove clinical causality.

Therefore, the explanations should be interpreted as model explanations, not as direct evidence of biological or medical mechanisms.
