import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import shap
from lime.lime_tabular import LimeTabularExplainer


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="SHAP and LIME Demo",
    layout="wide"
)

st.title("SHAP and LIME Demo")
st.write(
    "This app demonstrates explainable AI using the Breast Cancer Wisconsin dataset. "
    "A Random Forest model is trained, and its predictions are explained using SHAP and LIME."
)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

target_names = data.target_names
feature_names = data.feature_names

st.subheader("Dataset information")

st.write(f"Number of samples: **{X.shape[0]}**")
st.write(f"Number of features: **{X.shape[1]}**")
st.write(f"Target classes: **0 = {target_names[0]}**, **1 = {target_names[1]}**")

with st.expander("View dataset"):
    df = X.copy()
    df["target"] = y
    df["target_name"] = df["target"].map({
        0: target_names[0],
        1: target_names[1]
    })
    st.dataframe(df)


# ---------------------------------------------------------
# Train/test split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# ---------------------------------------------------------
# Train model
# ---------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model")
st.write("Model used: **Random Forest Classifier**")
st.write(f"Test accuracy: **{accuracy:.3f}**")


# ---------------------------------------------------------
# Select individual case
# ---------------------------------------------------------

st.subheader("Select a case to explain")

X_test_display = X_test.reset_index(drop=True)
y_test_display = y_test.reset_index(drop=True)

case_index = st.slider(
    "Choose a test case",
    min_value=0,
    max_value=len(X_test_display) - 1,
    value=0
)

selected_case = X_test_display.iloc[[case_index]]
true_class = y_test_display.iloc[case_index]

predicted_class = model.predict(selected_case)[0]
predicted_probabilities = model.predict_proba(selected_case)[0]

st.write(f"True class: **{true_class} — {target_names[true_class]}**")
st.write(f"Predicted class: **{predicted_class} — {target_names[predicted_class]}**")

prob_df = pd.DataFrame({
    "Class": target_names,
    "Probability": predicted_probabilities
})

st.write("Predicted probabilities:")
st.dataframe(prob_df)


# ---------------------------------------------------------
# SHAP explanations
# ---------------------------------------------------------

st.header("SHAP explanation")

st.write(
    "SHAP estimates how much each feature contributes to a model prediction. "
    "Positive values push the prediction towards the selected class; negative values push it away."
)

explainer_shap = shap.TreeExplainer(model)
shap_values = explainer_shap.shap_values(X_test_display)


def get_shap_values_for_class(shap_values, class_index):
    """
    Handles different SHAP output formats.
    Some versions return a list, others return a 3D array.
    """

    if isinstance(shap_values, list):
        return shap_values[class_index]

    if isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            return shap_values[:, :, class_index]
        elif shap_values.ndim == 2:
            return shap_values

    raise ValueError("Unexpected SHAP values format.")


selected_class_for_explanation = st.selectbox(
    "Choose class to explain with SHAP",
    options=[0, 1],
    format_func=lambda x: f"{x} — {target_names[x]}",
    index=int(predicted_class)
)

shap_class_values = get_shap_values_for_class(
    shap_values,
    selected_class_for_explanation
)


# ---------------------------------------------------------
# SHAP global importance
# ---------------------------------------------------------

st.subheader("SHAP global explanation")

st.write(
    "This plot shows the average absolute SHAP value for each feature. "
    "Features with higher values had a stronger overall influence on the model."
)

mean_abs_shap = np.abs(shap_class_values).mean(axis=0)

shap_importance = pd.DataFrame({
    "Feature": feature_names,
    "Mean absolute SHAP value": mean_abs_shap
}).sort_values(
    by="Mean absolute SHAP value",
    ascending=False
).head(15)

fig_global, ax_global = plt.subplots(figsize=(8, 6))

ax_global.barh(
    shap_importance["Feature"][::-1],
    shap_importance["Mean absolute SHAP value"][::-1]
)

ax_global.set_xlabel("Mean absolute SHAP value")
ax_global.set_title(
    f"Global SHAP importance for class: {target_names[selected_class_for_explanation]}"
)

st.pyplot(fig_global)


# ---------------------------------------------------------
# SHAP local explanation
# ---------------------------------------------------------

st.subheader("SHAP local explanation")

st.write(
    "This plot explains one selected prediction. "
    "It shows which features contributed most to the prediction for the selected case."
)

local_shap_values = shap_class_values[case_index]
local_feature_values = selected_case.iloc[0]

local_shap_df = pd.DataFrame({
    "Feature": feature_names,
    "Feature value": local_feature_values.values,
    "SHAP value": local_shap_values
})

local_shap_df["Absolute SHAP value"] = np.abs(local_shap_df["SHAP value"])

local_shap_df = local_shap_df.sort_values(
    by="Absolute SHAP value",
    ascending=False
).head(10)

fig_local, ax_local = plt.subplots(figsize=(8, 6))

ax_local.barh(
    local_shap_df["Feature"][::-1],
    local_shap_df["SHAP value"][::-1]
)

ax_local.axvline(0, linestyle="--")
ax_local.set_xlabel("SHAP value")
ax_local.set_title(
    f"Local SHAP explanation for case {case_index}, class: {target_names[selected_class_for_explanation]}"
)

st.pyplot(fig_local)

with st.expander("View SHAP values for selected case"):
    st.dataframe(local_shap_df)


# ---------------------------------------------------------
# LIME explanation
# ---------------------------------------------------------

st.header("LIME explanation")

st.write(
    "LIME explains one individual prediction by creating small perturbations around the selected case "
    "and fitting a simple local model to approximate the behaviour of the original model."
)

lime_explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=list(feature_names),
    class_names=list(target_names),
    mode="classification",
    discretize_continuous=True,
    random_state=42
)

selected_class_for_lime = st.selectbox(
    "Choose class to explain with LIME",
    options=[0, 1],
    format_func=lambda x: f"{x} — {target_names[x]}",
    index=int(predicted_class)
)

lime_exp = lime_explainer.explain_instance(
    data_row=selected_case.iloc[0].values,
    predict_fn=model.predict_proba,
    num_features=10,
    labels=[selected_class_for_lime]
)

st.subheader("LIME local explanation")

st.write(
    f"LIME explanation for case **{case_index}**, "
    f"class **{selected_class_for_lime} — {target_names[selected_class_for_lime]}**."
)

components.html(
    lime_exp.as_html(labels=[selected_class_for_lime]),
    height=800,
    scrolling=True
)


# ---------------------------------------------------------
# Notes
# ---------------------------------------------------------

st.header("Interpretation notes")

st.info(
    "SHAP and LIME explain the behaviour of the model, not necessarily the real biological or clinical mechanism. "
    "They should be interpreted as model explanations, not as causal explanations."
)

st.markdown(
    """
    **Scientific references**

    Ribeiro, Marco Tulio, Sameer Singh, and Carlos Guestrin (2016). 
    “Why Should I Trust You? Explaining the Predictions of Any Classifier.” 
    *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 1135–1144.

    Lundberg, Scott M., and Su-In Lee (2017). 
    “A Unified Approach to Interpreting Model Predictions.” 
    *Advances in Neural Information Processing Systems*, 30.
    """
)
