import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
import plotly.express as px
import yagmail

file=0

def risk(prob):
    if prob<=0.36:
        return 0
    if prob>0.36 and prob<=0.66:
        return 1
    if prob>0.66:
        return 2
    

def dropout(predictions):
    mapping={
        0:"LOW",
        1:'MEDIUM',
        2:"HIGH"
    }
    
    out=[]
    for prob in predictions: 
       
        out.append(mapping[risk(prob)]) 
    
    return np.array(out)

warnings.filterwarnings('ignore')
def predict_dropout(df):
    """
    Predicts dropout probability for each student using Random Forest.
    Assumes 'Target' column exists for training 
    """
    with open("model.pkl", "rb") as f:
        loaded_model = pickle.load(f)
    roll = df.iloc[:,0:1].values
    email = df.iloc[:,-1:].values
    ndf=df.iloc[:,1:-1].values
    p = []
    for i in ndf:
        pro = loaded_model.predict_proba(i.reshape(1,-1))[0][0]
        p.append(pro)
   
    return np.array(p),np.array(roll),np.array(email) 


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
        df = pd.read_csv(uploaded_file)
        st.session_state['data'] = df
        st.session_state['filename'] = uploaded_file.name
        
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        
        
        
        # Show data preview
        # st.subheader("Data Preview")
        # st.dataframe(df.head(10), use_container_width=True)
        
        
        
        
        
      
        # Student dropout prediction section
        st.subheader("Dropout Risk Prediction")
        st.info("""
        **Assumptions:** 
        - The CSV must have a 'Target' column with values like 'Dropout', 'Graduate' (or similar; mapped to binary: Dropout=1, others=0).
        - Features include categorical (e.g., Marital status, Gender) and numerical (e.g., Age, Grades) columns.
        - The model uses Random Forest Classifier trained on 80% of data and predicts probabilities for all students.
        - High risk if probability > 0.5.
        """)
        
        if st.button("Predict Dropout Risks"):
            with st.spinner("Training model and predicting..."):
                # Backend processing for dropout prediction
                predictions,roll,email = predict_dropout(df)
                risk_values = dropout(predictions)  # Apply your dropout mapping function

                # Create DataFrame
                risktype = pd.DataFrame({
                    "Roll": roll.ravel(),
                    "Mentor_Email": email.ravel(),
                    "RiskType": risk_values
                })
                st.session_state['predictions'] = risktype
                # Display in Streamlit
                st.success("Predictions completed!")
                st.dataframe(risktype, use_container_width=True)
                   

    
                counts = pd.Series(predictions).value_counts()

                # Get count of students
                high_risk_count = (risktype.iloc[:,-1]=='HIGH').sum()
                low_risk_count = (risktype.iloc[:,-1]=='LOW').sum()
                mid_risk_count = (risktype.iloc[:,-1]=='MEDIUM').sum()
                
                # show number for students of each type
                st.metric("High Risk Students", high_risk_count)
                st.metric("Medium Risk Students", mid_risk_count)
                st.metric("Low Risk Students", low_risk_count)
                
                st.subheader("Dropout Risk Distribution 📊")

                # Prepare the data into a Pandas DataFrame for Plotly
                risk_data = pd.DataFrame({
                    'Risk Level': ['High', 'Medium', 'Low'],
                    'Student Count': [high_risk_count, mid_risk_count, low_risk_count]
                })

                #color mapping
                color_map = {'High': 'red', 'Medium': 'orange', 'Low': 'green'}

                col_chart, col_pie = st.columns(2)

                # bar graph
                with col_chart:
                    st.markdown("#### Bar Chart: Student Count by Risk Level")
                    bar_fig = px.bar(
                        risk_data,
                        x='Risk Level',
                        y='Student Count',
                        title='Risk Level Counts',
                        color='Risk Level',
                        color_discrete_map=color_map,
                        labels={'Student Count': 'Number of Students'}
                    )
                    # Ensure High is at the right for ascending risk
                    bar_fig.update_xaxes(categoryorder='array', categoryarray=['Low', 'Medium', 'High'])
                    st.plotly_chart(bar_fig, use_container_width=True)

                #  Pie Chart 
                with col_pie:
                    st.markdown("#### Pie Chart: Risk Percentage Breakdown")
                    pie_fig = px.pie(
                        risk_data,
                        values='Student Count',
                        names='Risk Level',
                        title='Percentage Breakdown',
                        color='Risk Level',
                        color_discrete_map=color_map,
                        hole=0.3 # Adds a donut shape
                    )
                    st.plotly_chart(pie_fig, use_container_width=True)
                    
                    ##mail
                    df = risktype
                    df = df[df['RiskType'] == 'HIGH']

                    for i in range(len(df)):
                        rol = df.iloc[i]['Roll']  
                        email = df.iloc[i]['Mentor_Email']
                        print(rol, email)

                        yag = yagmail.SMTP("devsoni14862@gmail.com", "btzs edou lddb kzwo")
                        yag.send(
                            to=email,
                            subject="Student report",
                            contents=f"STudent with roll no. {rol} has high chances of dropout! "
                        )
                        print("Alert email sent with yagmail!")
       
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
            st.info("Run predictions first to enable download.")

        
 
else:
    st.info("Please upload a CSV file with student data to get started")


if 'data' in st.session_state:
    st.sidebar.success("Student data is available!")
    st.sidebar.write(f"File: {st.session_state.get('filename', 'N/A')}")
    st.sidebar.write(f"Shape: {st.session_state['data'].shape}")
    








