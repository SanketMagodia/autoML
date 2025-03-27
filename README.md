# 🤖 AutoML Pipeline: Intelligent Data Transformation & Model Training

## 📝 Project Overview

This AutoML Pipeline is an intelligent data transformation and machine learning solution that leverages AI agents to automatically preprocess, transform, and train models on your CSV datasets. Key features include:

- 🧠 LLM-Powered Data Transformation
- 🤝 Collaborative AI Agents (Assistant & User Proxy)
- 📊 Automatic Model Training & Evaluation
- 🔍 MLflow Experiment Tracking
- 📈 Streamlit Interactive Dashboard

## 🚀 Project Components

### Core Technologies
- AutoGen: For intelligent code generation
- MLflow: For experiment tracking
- Scikit-learn: For machine learning pipelines
- Streamlit: For interactive web application
- Pandas: For data manipulation

## 🛠 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Clone the Repository
```bash
git clone https://github.com/SanketMagodia/autoML.git
cd autoML
```

### Install Dependencies
```bash
pip install -r req.txt
```

## 📦 Required Dependencies
- streamlit
- mlflow
- autogen
- scikit-learn
- pandas
- numpy

## 🌟 Features

### 1. Intelligent Data Transformation
- Automatic missing value handling
- Feature scaling
- Categorical encoding
- LLM-guided transformation suggestions

### 2. Automatic Model Training
- Classification and Regression support
- Multiple model training (Logistic Regression, Random Forest, SVM)
- Comprehensive performance metrics

### 3. Experiment Tracking
- MLflow integration
- Model comparison
- Performance logging

## 🖥 Running the Streamlit App

```bash
streamlit run app.py
```
## 🖥 Running on CLI 

```bash
python autoML.py --path "D:\project\AutoMl\Student_Mental_Stress_and_Coping_Mechanisms.csv" --project "Student stress" --target_variable "Substance Use" --problem "R"
```
### App Workflow
1. 📤 Upload CSV File
2. 🎯 Select Target Variable
3. 🤖 AI-Powered Transformation
4. 📊 View Original vs Transformed Data
5. 🔬 Explore Model Training Results

## 🔍 How It Works

### Data Transformation Process
1. LLM Agent analyzes dataset
2. Generates transformation code
3. User Proxy Agent validates and tests code
4. Iterative refinement of transformation strategy

### Model Training
- Automatic problem type detection
- Training multiple algorithms
- Performance metric logging in MLflow


## 🧪 Example Use Cases
- Sales Prediction
- Customer Churn Analysis
- Medical Diagnostics
- Financial Forecasting

## 🛡 Limitations & Considerations
- Works best with structured CSV data
- Limited to tabular datasets
- Transformation complexity depends on LLM's understanding

## 🤝 Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License
MIT License

## 🙌 Acknowledgements
- AutoGen Team
- MLflow
- Streamlit Community



---

**Disclaimer**: AI-generated transformations should always be reviewed by domain experts before production use.
