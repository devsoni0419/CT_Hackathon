import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Student Dropout Prediction",
    page_icon="🎓",
    layout="wide"
)

# Title
st.title("Student Data Upload & Dropout Prediction")
st.write("Upload your student data CSV file to analyze and predict dropout risks")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file with student data", type=['csv'])

if uploaded_file is not None:
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        # Store in session state for backend access
        st.session_state['data'] = df
        st.session_state['filename'] = uploaded_file.name
        
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        
        # Display file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Students", len(df))
        with col2:
            st.metric("Total Features", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        # Show data preview
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        
        
        # Backend processing section
        st.subheader("Backend Processing Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Show Basic Statistics"):
                st.write("**Numerical Features Statistics:**")
                st.dataframe(df.describe(), use_container_width=True)
        
        with col2:
            if st.button("Show Data Info"):
                buffer = []
                buffer.append(f"Shape: {df.shape}")
                buffer.append(f"Columns: {', '.join(df.columns.tolist())}")
                buffer.append(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
                st.text("\n".join(buffer))
        
        # Student dropout prediction section
        st.subheader("Dropout Risk Prediction")
        st.info("""
        **Assumptions:** 
        - The CSV must have a 'Target' column with values like 'Dropout', 'Enrolled', 'Graduate' (or similar; mapped to binary: Dropout=1, others=0).
        - Features include categorical (e.g., Marital status, Gender) and numerical (e.g., Age, Grades) columns.
        - The model uses Random Forest Classifier trained on 80% of data and predicts probabilities for all students.
        - High risk if probability > 0.5.
        """)
        
        if st.button("Predict Dropout Risks"):
            with st.spinner("Training model and predicting..."):
                # Backend processing for dropout prediction
                predictions = predict_dropout(df)
                
                # Display predictions
                st.success("Predictions completed!")
                st.dataframe(predictions, use_container_width=True)
                
                # Summary stats
                high_risk_count = (predictions['Dropout_Probability'] > 0.5).sum()
                st.metric("High Risk Students (Prob > 0.5)", high_risk_count)
        
        # Download processed data with predictions
        st.subheader("Download Data with Predictions")
        if 'predictions' in st.session_state:
            csv = st.session_state['predictions'].to_csv(index=False)
            st.download_button(
                label="Download CSV with Predictions",
                data=csv,
                file_name="student_dropout_predictions.csv",
                mime="text/csv"
            )
        else:
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Original CSV",
                data=csv,
                file_name="student_data.csv",
                mime="text/csv"
            )
        
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")

else:
    st.info("Please upload a CSV file with student data to get started")


# Backend function for dropout prediction
def predict_dropout(df):
    """
    Predicts dropout probability for each student using Random Forest.
    Assumes 'Target' column exists for training (Dropout=1, others=0).
    """
    if 'Target' not in df.columns:
        raise ValueError("CSV must contain a 'Target' column (e.g., 'Dropout', 'Enrolled', 'Graduate')")
    
    # Map target to binary: Dropout=1, others=0
    df['Target_Binary'] = df['Target'].apply(lambda x: 1 if 'Dropout' in str(x).upper() else 0)
    
    # Separate features and target
    X = df.drop(['Target', 'Target_Binary'], axis=1, errors='ignore')
    y = df['Target_Binary']
    
    # Handle categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict probabilities on full data
    dropout_probs = model.predict_proba(X)[:, 1]
    
    # Add predictions to original df
    df_result = df.copy()
    df_result['Dropout_Probability'] = dropout_probs
    df_result['Risk_Level'] = df_result['Dropout_Probability'].apply(
        lambda p: 'High' if p > 0.5 else 'Low' if p < 0.3 else 'Medium'
    )
    
    # Store in session state
    st.session_state['predictions'] = df_result
    
    # Optional: Model performance on test set
    y_pred = model.predict(X_test)
    st.info(f"Model Accuracy on Test Set: {model.score(X_test, y_test):.2f}")
    
    return df_result


# Access data from session state (backend access example)
if 'data' in st.session_state:
    st.sidebar.success("Student data is available!")
    st.sidebar.write(f"File: {st.session_state.get('filename', 'N/A')}")
    st.sidebar.write(f"Shape: {st.session_state['data'].shape}")