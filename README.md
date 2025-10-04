# CT_Hackathon
## Team CodeXplore (T015)
## Team members:
Dev Soni(Team Leader),
Sunny Yadav,
Rishi Bisht,
Mukesh Yadav,
Vivek Sharma

# Student Dropout Prediction System 🎓

A Streamlit-based web application that predicts student dropout risks using machine learning and automatically sends email alerts to mentors for high-risk students.

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

## Setup

### 1. Model File

Ensure you have a trained model file named `model.pkl` in the same directory as the application. This file should contain a trained Random Forest classifier.

### 2. Email Configuration

Update the email credentials in the code (line with `yagmail.SMTP`):

```python
yag = yagmail.SMTP("your_email@gmail.com", "your_app_password")
```

**Important**: Use Gmail App Password, not your regular password:
1. Enable 2-Factor Authentication on your Google account
2. Generate an App Password at: https://myaccount.google.com/apppasswords
3. Use the generated 16-character password in the code

## Usage

### 1. Start the Application

```bash
streamlit run app.py
```

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





