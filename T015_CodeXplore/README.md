# Team CodeXplore (T-015)
## Team Members:
Dev Soni(Team Leader) (Backend + Frontend)

Rishi Bisht (Frontend + Backend)

Sunny Yadav (Data Processing+Backend)

Mukesh Yadav (Data Acquisition + Data Processing)

Vivek Sharma (Frontend)

# Student Dropout Prediction System 🎓

A Streamlit-based web application that predicts student dropout risks using machine learning and automatically sends email alerts to mentors for high-risk students.

## Problem Statement

### Data Science & AI for Social Good: AI-based Drop-out Prediction and Counseling System

**Background:**  
By the time term-end marks reveal failures, many struggling students have disengaged beyond recovery. Counsellors and mentors need a mechanism that surfaces risk indicators—failing attendance, high number of attempts exhausted to pass a particular subject, reducing test scores—weeks earlier.

**The Challenge:**  
Attendance percentages live in one spreadsheet, test results in another, and fee-payment data in a third. No single view exists to signal that a learner is slipping in multiple areas simultaneously. Commercial analytics platforms promise predictive insights but demand funds and maintenance beyond the reach of public institutes. 

A simpler, transparent approach would merge existing spreadsheets, apply clear logic to colour-code risk, and notify mentors on a predictable schedule. Such a system must be easy to configure, require minimal training, and empower educators—not replace their judgment. 

By focusing on data fusion and timely alerts rather than complex algorithms, the institute can intervene early and reduce drop-out rates without fresh budget lines. This challenge epitomises the hackathon spirit: take what is already present, integrate it cleverly, and produce meaningful impact using machine learning.

**Expected Solution:**  
A consolidated digital dashboard that automatically ingests attendance, assessment scores, and other student-related data; applies clear, rule-based thresholds to identify at-risk students; highlights them in an intuitive visual format; and dispatches regular notifications to mentors and guardians, ensuring early, data-driven intervention achieved entirely through suitable machine learning approaches.

## Features

- **CSV Upload**: Easy file upload interface for student data
- **Dropout Prediction**: ML-powered risk assessment using Random Forest classifier
- **Risk Classification**: Students categorized into LOW, MEDIUM, and HIGH risk levels
- **Interactive Visualizations**: Bar charts and pie charts showing risk distribution
- **Automated Email Alerts**: Automatic notifications to mentors for high-risk students
- **Data Export**: Download predictions as CSV file

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Required Packages

Install all dependencies using:

```bash
pip install streamlit pandas numpy plotly yagmail
```

Or create a `requirements.txt` file:

```
streamlit
pandas
numpy
plotly
yagmail
```

Then install with:

```bash
pip install -r requirements.txt
```

## Project Structure

```
project/
│
├── app.py                 # Main Streamlit application
├── model.pkl             # Trained Random Forest model (required)
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Quick Start Guide

### Step 1: Clone or Download the Project

```bash
# If using git
git clone <repository-url>
cd student-dropout-prediction

# Or download and extract the ZIP file
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

If you don't have a requirements.txt file, install manually:

```bash
pip install streamlit pandas numpy plotly yagmail scikit-learn
```

### Step 4: Configure Email Settings

Open `app.py` and update the email configuration (around line 155):

```python
yag = yagmail.SMTP("your_email@gmail.com", "your_app_password")
```

**Setting up Gmail App Password:**

1. Go to your Google Account: https://myaccount.google.com
2. Select **Security** from the left menu
3. Enable **2-Step Verification** (if not already enabled)
4. Go to **App Passwords**: https://myaccount.google.com/apppasswords
5. Select app: **Mail**
6. Select device: **Other (Custom name)** → Enter "Student Dropout System"
7. Click **Generate**
8. Copy the 16-character password (format: `xxxx xxxx xxxx xxxx`)
9. Paste it in the code (remove spaces): `"your_app_password"`

### Step 5: Add the Model File

Ensure `model.pkl` is in the project root directory:

```
project/
├── app.py
├── model.pkl          ← Required file
├── requirements.txt
└── README.md
```

If you need to train a model, you'll need a training script (not included). The model should be a trained scikit-learn Random Forest classifier saved using pickle.

### Step 6: Run the Application

```bash
streamlit run app.py
```

The application will open automatically in your default browser at `http://localhost:8501`

If it doesn't open automatically, navigate to the URL shown in your terminal.

## Usage

### 1. Access the Application

Open your browser and go to `http://localhost:8501`

### 2. Upload Student Data

The CSV file must include:
- **First column**: Student Roll Number
- **Last column**: Mentor Email Address
- **Middle columns**: Feature data for prediction (grades, attendance, demographics, etc.)

### 3. Generate Predictions

Click the "Predict Dropout Risks" button to:
- Generate dropout probability for each student
- Classify students into risk categories
- View interactive visualizations
- Automatically send alerts to mentors of high-risk students

### 4. Download Results

Download the predictions as a CSV file containing:
- Roll Number
- Mentor Email
- Risk Type (LOW/MEDIUM/HIGH)

## Risk Classification

Students are classified based on dropout probability:

| Risk Level | Probability Range | Action |
|------------|------------------|---------|
| **LOW** | ≤ 0.36 | Monitor regularly |
| **MEDIUM** | 0.36 - 0.66 | Increased monitoring |
| **HIGH** | > 0.66 | Immediate intervention + Email alert |

## Email Alerts

The system automatically sends email notifications to mentors when their students are classified as HIGH risk. The email includes:
- Subject: "Student report"
- Content: Student roll number and dropout warning

## Data Privacy

⚠️ **Important Security Notes**:
- Never commit email credentials to version control
- Use environment variables for sensitive information
- Consider implementing OAuth for production use
- Ensure compliance with data protection regulations (GDPR, FERPA, etc.)

## Troubleshooting

### Common Issues

**"model.pkl not found"**
- Ensure the model file is in the same directory as app.py

**Email not sending**
- Verify Gmail App Password is correct
- Check internet connection
- Ensure "Less secure app access" is not blocking (use App Password instead)

**Import errors**
- Install all required packages: `pip install -r requirements.txt`

**Prediction errors**
- Verify CSV format matches expected structure
- Check for missing values in feature columns

## Future Enhancements

- [ ] Add user authentication
- [ ] Support multiple model types
- [ ] Implement email templates
- [ ] Add historical tracking
- [ ] Create admin dashboard
- [ ] Add model retraining capability
