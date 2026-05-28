import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Similar Cases Demo",
    layout="wide"
)

st.title("Similar Cases Demo")
st.write(
    "This app demonstrates an example-based explanation approach using the "
    "Breast Cancer Wisconsin dataset. Instead of explaining a prediction with "
    "SHAP or LIME, the app shows training cases that are most similar to the "
    "selected test case."
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
# Standardise data for similarity calculation
# ---------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ---------------------------------------------------------
# Select case
# ---------------------------------------------------------

st.subheader("Select a case to analyse")

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

# Combine train and selected case for a common PCA projection
combined_scaled = np.vstack([X_train_scaled, selected_case_scaled])

pca = PCA(n_components=2, random_state=42)
combined_pca = pca.fit_transform(combined_scaled)

X_train_pca = combined_pca[:-1]
selected_case_pca = combined_pca[-1]

similar_indices = indices[0]
similar_cases_pca = X_train_pca[similar_indices]

fig, ax = plt.subplots(figsize=(8, 6))

# Plot all training cases
for class_value in sorted(y_train.unique()):
    class_mask = y_train.values == class_value

    ax.scatter(
        X_train_pca[class_mask, 0],
        X_train_pca[class_mask, 1],
        alpha=0.25,
        label=f"Training class {class_value} — {target_names[class_value]}"
    )

# Plot similar cases
ax.scatter(
    similar_cases_pca[:, 0],
    similar_cases_pca[:, 1],
    s=120,
    marker="o",
    edgecolors="black",
    linewidths=1.5,
    label="Similar cases"
)

# Plot selected case
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
# Interpretation notes
# ---------------------------------------------------------

st.header("Interpretation notes")

st.info(
    "Similar cases explanations are intuitive, but they depend strongly on how "
    "similarity is defined. Here, similarity is based on Euclidean distance after "
    "standardising all features. This does not prove clinical equivalence or causality."
)

st.markdown(
    """
    **Key idea**

    Instead of asking only “Which features influenced the model?”, this approach asks:

    **“Which previous cases look most similar to this case?”**

    This can be useful in teaching because students can inspect concrete examples
    rather than only abstract feature contributions.
    """
)

st.markdown(
    """
    **Scientific references**

    Aamodt, Agnar, and Enric Plaza (1994). “Case-Based Reasoning: Foundational Issues, Methodological Variations, and System Approaches.” *AI Communications*, 7(1), 39–59.

    Kim, Been, Cynthia Rudin, and Julie Shah (2014). “The Bayesian Case Model: A Generative Approach for Case-Based Reasoning and Prototype Classification.” *Advances in Neural Information Processing Systems*, 27.

    Caruana, Rich, Hooshang Kangarloo, John David Dionisio, Usha Sinha, and David Johnson (1999). “Case-Based Explanation of Non-Case-Based Learning Methods.” *Proceedings of the AMIA Symposium*, 212–215.
    """
)
