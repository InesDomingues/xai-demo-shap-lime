import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Similar Cases + Decision Tree Demo",
    layout="wide"
)

st.title("Similar Cases + Decision Tree Demo")

st.write(
    "This app demonstrates an example-based explanation approach using the "
    "Breast Cancer Wisconsin dataset. A Decision Tree classifier is trained, "
    "and the app shows both the tree structure and the most similar training cases "
    "to a selected test case."
)


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

target_names = data.target_names
feature_names = data.feature_names

df = X.copy()
df["target"] = y
df["target_name"] = df["target"].map({
    0: target_names[0],
    1: target_names[1]
})


st.subheader("Dataset information")

st.write(f"Number of samples: **{X.shape[0]}**")
st.write(f"Number of features: **{X.shape[1]}**")
st.write(f"Target classes: **0 = {target_names[0]}**, **1 = {target_names[1]}**")

with st.expander("View full dataset"):
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
# Decision Tree settings
# ---------------------------------------------------------

st.subheader("Decision Tree settings")

max_depth = st.slider(
    "Choose the maximum depth of the tree",
    min_value=1,
    max_value=6,
    value=3
)

min_samples_split = st.slider(
    "Choose the minimum number of samples required to split",
    min_value=2,
    max_value=20,
    value=5
)

min_samples_leaf = st.slider(
    "Choose the minimum number of samples in a leaf",
    min_value=1,
    max_value=10,
    value=2
)


# ---------------------------------------------------------
# Train Decision Tree model
# ---------------------------------------------------------

model = DecisionTreeClassifier(
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


st.subheader("Model")

st.write("Model used: **Decision Tree Classifier**")
st.write(f"Test accuracy: **{accuracy:.3f}**")


# ---------------------------------------------------------
# Draw Decision Tree with custom colours
# ---------------------------------------------------------

st.header("Decision Tree")

st.write(
    "This figure shows the structure of the trained decision tree. "
    "Malignant nodes are shown in red and benign nodes are shown in green."
)

fig_tree, ax_tree = plt.subplots(figsize=(24, 12))

artists = plot_tree(
    model,
    feature_names=feature_names,
    class_names=target_names,
    filled=False,
    rounded=True,
    fontsize=8,
    ax=ax_tree
)

# Custom node colours:
# malignant -> red
# benign -> green
for artist in artists:
    text = artist.get_text()
    bbox = artist.get_bbox_patch()

    if bbox is not None:
        if "class = malignant" in text:
            bbox.set_facecolor("#f4a6a6")  # light red
        elif "class = benign" in text:
            bbox.set_facecolor("#b7e4c7")  # light green
        else:
            bbox.set_facecolor("#eeeeee")

        bbox.set_edgecolor("black")
        bbox.set_alpha(0.95)

ax_tree.set_title("Decision Tree: malignant = red, benign = green", fontsize=16)

st.pyplot(fig_tree)


# ---------------------------------------------------------
# Standardise data for similarity calculation
# ---------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ---------------------------------------------------------
# Select case
# ---------------------------------------------------------

st.header("Select a case to analyse")

X_test_display = X_test.reset_index(drop=True)
y_test_display = y_test.reset_index(drop=True)

case_index = st.slider(
    "Choose a test case",
    min_value=0,
    max_value=len(X_test_display) - 1,
    value=0
)

selected_case = X_test_display.iloc[[case_index]]
selected_case_scaled = X_test_scaled[case_index].reshape(1, -1)

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

with st.expander("View selected case values"):
    selected_case_display = selected_case.copy()
    selected_case_display["true_class"] = true_class
    selected_case_display["predicted_class"] = predicted_class
    st.dataframe(selected_case_display)


# ---------------------------------------------------------
# Similar cases
# ---------------------------------------------------------

st.header("Similar cases explanation")

st.write(
    "The app finds the most similar training cases by calculating distances "
    "between the selected case and all training cases. Features are standardised "
    "before calculating similarity, so variables with larger numerical scales "
    "do not dominate the distance calculation."
)

k = st.slider(
    "Number of similar cases to show",
    min_value=3,
    max_value=15,
    value=5
)

nearest_neighbors = NearestNeighbors(
    n_neighbors=k,
    metric="euclidean"
)

nearest_neighbors.fit(X_train_scaled)

distances, indices = nearest_neighbors.kneighbors(selected_case_scaled)

similar_X = X_train.iloc[indices[0]].copy()
similar_y = y_train.iloc[indices[0]].copy()

similar_cases = similar_X.copy()
similar_cases["target"] = similar_y.values
similar_cases["target_name"] = similar_cases["target"].map({
    0: target_names[0],
    1: target_names[1]
})
similar_cases["distance_to_selected_case"] = distances[0]

similar_cases = similar_cases.reset_index(drop=True)


st.subheader("Most similar training cases")

st.dataframe(similar_cases)


# ---------------------------------------------------------
# Summary of similar cases
# ---------------------------------------------------------

st.subheader("Summary of similar cases")

similar_class_counts = similar_cases["target_name"].value_counts().reset_index()
similar_class_counts.columns = ["Class", "Number of similar cases"]

st.dataframe(similar_class_counts)

majority_class = similar_cases["target"].mode()[0]

st.write(
    f"Among the **{k}** most similar training cases, the most common class is "
    f"**{majority_class} — {target_names[majority_class]}**."
)


# ---------------------------------------------------------
# Visualisation using PCA
# ---------------------------------------------------------

st.header("Visualisation of selected case and similar cases")

st.write(
    "Because the dataset has many features, PCA is used only for visualisation. "
    "The similarity calculation above is performed using all standardised features."
)

combined_scaled = np.vstack([X_train_scaled, selected_case_scaled])

pca = PCA(n_components=2, random_state=42)
combined_pca = pca.fit_transform(combined_scaled)

X_train_pca = combined_pca[:-1]
selected_case_pca = combined_pca[-1]

similar_indices = indices[0]
similar_cases_pca = X_train_pca[similar_indices]

fig, ax = plt.subplots(figsize=(8, 6))

for class_value in sorted(y_train.unique()):
    class_mask = y_train.values == class_value

    if target_names[class_value] == "malignant":
        label = "Training class 0 — malignant"
    else:
        label = "Training class 1 — benign"

    ax.scatter(
        X_train_pca[class_mask, 0],
        X_train_pca[class_mask, 1],
        alpha=0.25,
        label=label
    )

ax.scatter(
    similar_cases_pca[:, 0],
    similar_cases_pca[:, 1],
    s=120,
    marker="o",
    edgecolors="black",
    linewidths=1.5,
    label="Similar cases"
)

ax.scatter(
    selected_case_pca[0],
    selected_case_pca[1],
    s=180,
    marker="X",
    edgecolors="black",
    linewidths=1.5,
    label="Selected case"
)

ax.set_title("Selected case and most similar cases")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.legend()
ax.grid(True, alpha=0.3)

st.pyplot(fig)


# ---------------------------------------------------------
# Feature importance
# ---------------------------------------------------------

st.header("Feature importance from the decision tree")

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.dataframe(importance_df)

top_importance_df = importance_df.head(10)

fig_imp, ax_imp = plt.subplots(figsize=(8, 5))

ax_imp.barh(
    top_importance_df["Feature"][::-1],
    top_importance_df["Importance"][::-1]
)

ax_imp.set_xlabel("Importance")
ax_imp.set_title("Top 10 feature importances")

st.pyplot(fig_imp)


# ---------------------------------------------------------
# Interpretation notes
# ---------------------------------------------------------

st.header("Interpretation notes")

st.markdown(
    """
    **Key idea**

    This app combines two types of interpretability:

    1. **Rule-based interpretation** through the decision tree structure  
    2. **Example-based interpretation** through similar cases

    **Important caution**

    The tree and the similar cases explain the behaviour of the model and the dataset representation.
    They should not be interpreted as direct clinical evidence or causal explanation.
    """
)
