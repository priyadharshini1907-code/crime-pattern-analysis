import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Crime Classifier (Random Forest)", layout="wide")
st.title(" Crime Type Classification using Random Forest")

uploaded_file = st.file_uploader(" Upload Cleaned Crime Data (.csv)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader(" Data Preview")
    st.dataframe(df.head())

    # Column cleanup
    df.columns = df.columns.str.strip().str.upper().str.replace(" ", "_")
    target_column = 'OFFENSE_CODE_GROUP'

    # Drop rows with missing target
    df = df.dropna(subset=[target_column])

    # Group rare classes
    min_count = 100
    value_counts = df[target_column].value_counts()
    common_classes = value_counts[value_counts >= min_count].index
    df[target_column] = df[target_column].apply(lambda x: x if x in common_classes else 'Other')

    # Drop irrelevant columns
    drop_columns = ['INCIDENT_NUMBER', 'OFFENSE_DESCRIPTION', 'OCCURRED_ON_DATE',
                    'TIME', 'STREETREET', 'LOCATION']
    df = df.drop(columns=[col for col in drop_columns if col in df.columns])

    # Split features and label
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode features
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Impute and scale
    X = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(X), columns=X.columns)
    y_encoded = LabelEncoder().fit_transform(y.astype(str))
    X_scaled = StandardScaler().fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=18,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    st.success(f" Model Accuracy: {accuracy * 100:.2f}%")

    # Classification Report
    st.subheader(" Classification Report")
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    st.code(classification_report(y_test, y_pred, zero_division=0))

    # Plot F1 Scores
    report_df = pd.DataFrame(report).transpose()
    f1_data = report_df.iloc[:-3]['f1-score'].sort_values()

    st.subheader("F1 Scores by Class")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    f1_data.plot(kind='barh', ax=ax1, color='skyblue')
    ax1.set_xlabel("F1 Score")
    ax1.set_title("F1 Score by Crime Type")
    st.pyplot(fig1)

    # Plot Confusion Matrix
    st.subheader(" Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_title("Confusion Matrix")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    st.pyplot(fig2)

    # Plot Feature Importances
    st.subheader("Feature Importances")
    importances = model.feature_importances_
    feat_names = df.drop(columns=[target_column]).columns
    indices = np.argsort(importances)[::-1]

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importances[indices][:15], y=feat_names[indices][:15], ax=ax3, palette="viridis")
    ax3.set_title("Top 15 Important Features")
    st.pyplot(fig3)

else:
    st.info(" Please upload a cleaned CSV file to begin.")
