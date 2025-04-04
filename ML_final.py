import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import time
import pickle
import joblib
import io

# Set page configuration
st.set_page_config(page_title="ML Model Trainer", layout="wide")

# Title and description
st.title("Machine Learning Model Trainer 📊🚀 ")
st.markdown("Select datasets, features, and models to train and evaluate machine learning models.")

# Add information about how to use the app
with st.expander("How to use this app"):
    st.markdown("""
    1. Select a dataset from the sidebar or upload your own CSV file
    2. Select the task type (classification or regression)
    3. Choose a target column appropriate for your task
    4. Select numerical and categorical features for your model
    5. Select a model type based on your task
    6. Configure model parameters in the sidebar
    7. Click 'Fit Model' to train and evaluate your model
    8. Explore the performance metrics and visualizations
    9. Download the trained model
    """)

# Sidebar for selections
st.sidebar.header("Configuration")

# Dataset selection
dataset_option = st.sidebar.selectbox(
    "Select Dataset",
    ["Titanic Dataset", "Tips Dataset"]
)

# Load selected dataset
@st.cache_data
def load_data(dataset_name):
    if dataset_name == "Titanic Dataset":
        data = sns.load_dataset("titanic")
        default_target = "survived"
    else:  # Tips dataset
        data = sns.load_dataset("tips")
        default_target = "tip"
    return data, default_target

# Load data based on selection
data, default_target = load_data(dataset_option)

# Custom dataset upload option
uploaded_file = st.sidebar.file_uploader("Or upload your own CSV file 📁", type=["csv"])
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.sidebar.success("Custom dataset loaded successfully!")
        default_target = data.columns[0]  # Just a default, will be selected by user
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")

# Task selection - now independent of dataset
task_type = st.sidebar.selectbox(
    "Select Task Type",
    ["classification", "regression"]
)

# Display dataset preview
st.subheader("Dataset Preview ")
st.dataframe(data.head())

# Basic dataset info
st.subheader("Dataset Information")
col1, col2, col3 = st.columns(3)
with col1:
    st.write(f"Number of rows: {data.shape[0]}")
    st.write(f"Number of columns: {data.shape[1]}")
with col2:
    st.write(f"Task type: {task_type.capitalize()}")
with col3:
    # Count and display missing values
    missing_values = data.isnull().sum().sum()
    st.write(f"Missing values: {missing_values}")
    if missing_values > 0:
        st.write("✓ Will be imputed automatically")

# Missing values handling options
imputation_method = "mean/mode"
if missing_values > 0:
    with st.expander("Missing Values Handling"):
        imputation_method = st.radio(
            "Select imputation strategy for missing values:",
            ["mean/mode", "median/mode", "drop rows"],
            index=0,
            help="Mean/mode uses mean for numerical values and most frequent value for categorical. Median/mode uses median for numerical values. Drop rows removes samples with any missing values."
        )
        
        # Show columns with missing values
        missing_cols = data.columns[data.isnull().any()].tolist()
        if missing_cols:
            st.write("Columns with missing values:")
            for col in missing_cols:
                miss_count = data[col].isnull().sum()
                miss_percent = (miss_count / len(data)) * 100
                st.write(f"- {col}: {miss_count} values ({miss_percent:.1f}%)")

# Select target column based on task type
if task_type == "classification":
    # Suggest categorical columns first for classification
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    # For classification, prioritize bool/int columns with few unique values
    potential_targets = [col for col in numerical_cols if len(data[col].unique()) <= 10]
    potential_targets.extend(categorical_cols)
    # Add remaining columns
    potential_targets.extend([col for col in numerical_cols if col not in potential_targets])
    
    if default_target not in potential_targets:
        default_target = potential_targets[0] if potential_targets else data.columns[0]
else:  # regression
    # Suggest numerical columns first for regression
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    potential_targets = numerical_cols
    
    if default_target not in potential_targets:
        default_target = potential_targets[0] if potential_targets else data.columns[0]

target_column = st.sidebar.selectbox("Select Target Column", potential_targets, index=potential_targets.index(default_target) if default_target in potential_targets else 0)

# Feature selection
st.sidebar.subheader("Feature Selection")

# Handle categorical and numerical features
categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Remove target from potential features
if target_column in categorical_cols:
    categorical_cols.remove(target_column)
if target_column in numerical_cols:
    numerical_cols.remove(target_column)

# Select numerical features
selected_numerical = st.sidebar.multiselect(
    "Select Numerical Features",
    numerical_cols,
    default=numerical_cols[:min(3, len(numerical_cols))]
)

# Select categorical features
selected_categorical = st.sidebar.multiselect(
    "Select Categorical Features",
    categorical_cols,
    default=categorical_cols[:min(2, len(categorical_cols))]
)

# Model selection based on task type
st.sidebar.subheader("Model Selection")
if task_type == "regression":
    model_option = st.sidebar.selectbox(
        "Select Regression Model",
        ["Linear Regression", "Random Forest Regressor"]
    )
else:  # classification
    model_option = st.sidebar.selectbox(
        "Select Classification Model",
        ["Logistic Regression", "Random Forest Classifier"]
    )

# Model parameters configuration
st.sidebar.subheader("Model Parameters")

# Common parameters
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
random_state = st.sidebar.number_input("Random State", 0, 100, 42)

# Model-specific parameters
if model_option == "Random Forest Regressor" or model_option == "Random Forest Classifier":
    n_estimators = st.sidebar.slider("Number of Estimators", 10, 200, 100, 10)
    max_depth = st.sidebar.slider("Max Depth", 1, 20, 10)
    min_samples_split = st.sidebar.slider("Min Samples Split", 2, 10, 2)
elif model_option == "Logistic Regression":
    C = st.sidebar.slider("Regularization Parameter (C)", 0.01, 10.0, 1.0, 0.01)
    max_iter = st.sidebar.slider("Maximum Iterations", 100, 1000, 100, 100)

# Prepare data for modeling with missing value handling
def prepare_data(data, target_column, selected_numerical, selected_categorical, imputation_method):
    # Copy the data to avoid modifying the original
    data_copy = data.copy()
    
    # Get the target variable first (before any imputation)
    y = data_copy[target_column].copy()
    
    # Handle missing data in the target variable
    if y.isnull().any():
        if imputation_method == "drop rows":
            valid_indices = y.notnull()
            y = y[valid_indices]
            data_copy = data_copy[valid_indices]
        else:
            if task_type == "classification" or target_column in categorical_cols:
                # For classification target, use mode imputation
                y = y.fillna(y.mode()[0])
            else:
                # For regression target, use mean or median
                if imputation_method == "mean/mode":
                    y = y.fillna(y.mean())
                else:  # median/mode
                    y = y.fillna(y.median())
    
    # Create X dataframe with selected features
    X = pd.DataFrame()
    
    # Handle numerical features
    if selected_numerical:
        X_num = data_copy[selected_numerical].copy()
        
        # Impute missing values in numerical features
        if imputation_method == "drop rows":
            # Dropping will be handled later
            pass
        else:
            for col in selected_numerical:
                if X_num[col].isnull().any():
                    if imputation_method == "mean/mode":
                        X_num[col] = X_num[col].fillna(X_num[col].mean())
                    else:  # median/mode
                        X_num[col] = X_num[col].fillna(X_num[col].median())
        
        X = X_num
    
    # Handle categorical features with encoding
    encoders = {}
    if selected_categorical:
        X_cat = data_copy[selected_categorical].copy()
        
        # Impute missing values in categorical features (always use mode)
        for col in selected_categorical:
            if X_cat[col].isnull().any():
                if imputation_method == "drop rows":
                    # Dropping will be handled later
                    pass
                else:
                    # For categorical, always use mode imputation
                    X_cat[col] = X_cat[col].fillna(X_cat[col].mode()[0])
        
        # Encode categorical features
        for col in selected_categorical:
            le = LabelEncoder()
            # Handle potential NaN values during encoding
            X_cat[col] = X_cat[col].astype(str)
            X[col] = le.fit_transform(X_cat[col])
            encoders[col] = le
    
    # For "drop rows" imputation strategy, drop any rows with missing values
    if imputation_method == "drop rows":
        # Identify rows with missing values
        missing_mask = X.isnull().any(axis=1)
        if missing_mask.any():
            # Keep only rows without missing values
            X = X[~missing_mask]
            y = y[~missing_mask]
    
    # Encode target if classification and target is categorical
    if task_type == "classification" and target_column not in numerical_cols:
        le_target = LabelEncoder()
        y = y.astype(str)  # Convert to string to handle any potential NaN values
        y = le_target.fit_transform(y)
        encoders['target'] = le_target
    
    return X, y, encoders

# Function to calculate feature importance for any model
def get_feature_importance(model, X, y, X_test, y_test):
    if hasattr(model, 'feature_importances_'):
        # For models like Random Forest that have feature_importances_ attribute
        importances = model.feature_importances_
    else:
        # For models like Linear Regression and Logistic Regression
        # Use permutation importance
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=random_state)
        importances = result.importances_mean
    
    # Create DataFrame with feature importances
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    return feature_importance

# Create a form for model training
with st.form(key='model_training_form'):
    st.subheader("Model Training")
    fit_button = st.form_submit_button(label='Fit Model')

# Only train the model if the fit button is pressed
if fit_button:
    if len(selected_numerical + selected_categorical) > 0:
        # Prepare data with missing value handling
        X, y, encoders = prepare_data(data, target_column, selected_numerical, selected_categorical, imputation_method)
        
        if len(X) == 0:
            st.error("After handling missing values, no data remains for training. Try a different imputation method.")
        else:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            
            # Show training progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Train model based on selection
            try:
                status_text.text('Training model...')
                progress_bar.progress(25)
                time.sleep(0.5)  # Simulate longer processing
                
                if model_option == "Linear Regression":
                    model = LinearRegression()
                elif model_option == "Random Forest Regressor":
                    model = RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        random_state=random_state
                    )
                elif model_option == "Logistic Regression":
                    model = LogisticRegression(
                        C=C,
                        max_iter=max_iter,
                        random_state=random_state
                    )
                else:  # Random Forest Classifier
                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        random_state=random_state
                    )
                
                progress_bar.progress(50)
                status_text.text('Fitting model to training data...')
                model.fit(X_train, y_train)
                progress_bar.progress(75)
                status_text.text('Evaluating model...')
                y_pred = model.predict(X_test)
                progress_bar.progress(100)
                status_text.text('Model training complete!')
                time.sleep(0.5)  # Simulate longer processing
                status_text.empty()
                progress_bar.empty()
                
                # Display metrics
                st.subheader("Model Performance Metrics")
                
                if task_type == "regression":
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, y_pred)
                    
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    metrics_col1.metric("Mean Squared Error", f"{mse:.4f}")
                    metrics_col2.metric("Root Mean Squared Error", f"{rmse:.4f}")
                    metrics_col3.metric("R² Score", f"{r2:.4f}")
                    
                    # Visualizations for regression
                    st.subheader("Visualization of Results")
                    
                    # Residual plot
                    fig_residual, ax_residual = plt.subplots(figsize=(10, 6))
                    residuals = y_test - y_pred
                    ax_residual.scatter(y_pred, residuals)
                    ax_residual.axhline(y=0, color='r', linestyle='-')
                    ax_residual.set_title('Residual Distribution')
                    ax_residual.set_xlabel('Predicted Values')
                    ax_residual.set_ylabel('Residuals')
                    st.pyplot(fig_residual)
                    
                else:  # Classification
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    
                    metrics_col1, metrics_col2 = st.columns(2)
                    metrics_col1.metric("Accuracy", f"{accuracy:.4f}")
                    metrics_col1.metric("Precision", f"{precision:.4f}")
                    metrics_col2.metric("Recall", f"{recall:.4f}")
                    metrics_col2.metric("F1 Score", f"{f1:.4f}")
                    
                    # Visualizations for classification
                    st.subheader("Visualization of Results")
                    
                    # Confusion Matrix
                    cm = confusion_matrix(y_test, y_pred)
                    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', ax=ax_cm)
                    ax_cm.set_title('Confusion Matrix')
                    ax_cm.set_xlabel('Predicted Labels')
                    ax_cm.set_ylabel('True Labels')
                    st.pyplot(fig_cm)
                    
                    # ROC Curve (only if binary classification)
                    if len(np.unique(y)) == 2:
                        # Get probability predictions
                        y_proba = model.predict_proba(X_test)[:, 1]
                        fpr, tpr, _ = roc_curve(y_test, y_proba)
                        roc_auc = auc(fpr, tpr)
                        
                        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
                        ax_roc.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
                        ax_roc.plot([0, 1], [0, 1], 'k--')
                        ax_roc.set_xlim([0.0, 1.0])
                        ax_roc.set_ylim([0.0, 1.05])
                        ax_roc.set_xlabel('False Positive Rate')
                        ax_roc.set_ylabel('True Positive Rate')
                        ax_roc.set_title('Receiver Operating Characteristic')
                        ax_roc.legend(loc="lower right")
                        st.pyplot(fig_roc)
                
                # Feature importance for ALL models - moved outside the if/else condition
                feature_importance = get_feature_importance(model, X, y, X_test, y_test)
                
                fig_importance, ax_importance = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax_importance)
                ax_importance.set_title('Feature Importance')
                st.pyplot(fig_importance)
                
                # Display data processing summary
                st.subheader("Data Processing Summary")
                st.write(f"- Imputation method: {imputation_method}")
                st.write(f"- Original dataset size: {len(data)} samples")
                st.write(f"- Processed dataset size: {len(X)} samples")
                if imputation_method == "drop rows" and len(X) < len(data):
                    st.write(f"- Rows removed due to missing values: {len(data) - len(X)}")
                st.write(f"- Training set: {len(X_train)} samples")
                st.write(f"- Test set: {len(X_test)} samples")
                
                # Simplified Model download section
                st.subheader("Download Trained Model")
                
                # Create download button with pickle format
                model_buffer = io.BytesIO()
                pickle.dump(model, model_buffer)
                model_buffer.seek(0)
                
                st.download_button(
                    label="Download Model",
                    data=model_buffer,
                    file_name=f"{model_option.replace(' ', '_').lower()}_model.pkl",
                    mime="application/octet-stream",
                    help="Download the trained model in pickle format"
                )
                
            except Exception as e:
                st.error(f"Error during model training: {e}")
    else:
        st.error("Please select at least one feature to train the model.")
else:
    st.info("Configure your parameters and click 'Fit Model' to train and evaluate.")