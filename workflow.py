import os
import pandas as pd
import numpy as np
import mlflow
import autogen
# from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import LabelEncoder
import textwrap
from datetime import datetime
from mlflow.models import infer_signature, validate_serving_input

from sklearn.metrics import (
    accuracy_score, 
    mean_squared_error, 
    r2_score, 
    classification_report, 
    mean_absolute_error
)



class AutoMLPipeline:
    def __init__(self, config_list, base_dir=None):
        """
        Initialize the AutoML Pipeline with language model configuration
        
        Args:
            config_list (list): Configuration for language models
        """
         # Set base directory 
        self.base_dir = base_dir or os.getcwd()
        self.transformation = ""
        # Ensure base directory exists
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Create subdirectories for different stages
        self.input_dir = os.path.join(self.base_dir, 'input')
        self.transformed_dir = os.path.join(self.base_dir, 'transformed')
        self.output_dir = os.path.join(self.base_dir, 'output')
        
        # Create these directories
        for dir_path in [self.input_dir, self.transformed_dir, self.output_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Initialize AutoGen agents
        self.assistant = autogen.AssistantAgent(
            name="ml_assistant", 
            system_message="You are an expert data science and machine learning assistant. Generate precise, executable Python code for data transformations.",
            llm_config={"config_list": config_list}
        )
        
        self.user_proxy = autogen.UserProxyAgent(
            name="UserProxy", 
            human_input_mode="NEVER",
            max_consecutive_auto_reply=2,
            code_execution_config={
                "work_dir": self.transformed_dir, 
                "use_docker": False
            }
        )
    def resolve_path(self, input_path):
        """
        Resolve the full path for the input file
        
        Args:
            input_path (str): Relative or absolute path to the input file
        
        Returns:
            str: Absolute path to the input file
        """
        # Convert to absolute path
        abs_path = os.path.abspath(os.path.join(self.base_dir, input_path))
        
        # Copy the file to input directory if it's not already there
        input_filename = os.path.basename(abs_path)
        destination_path = os.path.join(self.input_dir, input_filename)
        
        # Copy the file if source and destination are different
        if abs_path != destination_path:
            import shutil
            shutil.copy2(abs_path, destination_path)
        
        return destination_path
    def analyze_data(self, csv_path: str) -> dict:
        """
        Analyze the uploaded CSV data
        
        Args:
            csv_path (str): Path to the uploaded CSV file
        
        Returns:
            dict: Data analysis information
        """
        try:
            df = pd.read_csv(csv_path)
            
            analysis_info = {
                'data_shape': df.shape,
                'columns': list(df.columns),
                'data_types': dict(df.dtypes),
                'missing_values': df.isnull().sum().to_dict(),
                'unique_values': {col: df[col].nunique() for col in df.columns}
            }
            
            return analysis_info
        
        except Exception as e:
            print(f"Error analyzing data: {e}")
            return {}
    
    import re

    def generate_transformation_code(self, data_analysis: dict, target_variable: str, fileName: str) -> str:
        """
        Generate just the simplified transformation python code based on data analysis and nothing else just code.
        
        Args:
            data_analysis (dict): Analysis of the dataset
            target_variable (str): Target variable for the ML task
            fileName (str): CSV file path
        
        Returns:
            str: Simplified executable transformation code
        """
        transformation_request = f"""
        Generate just a simple Python function (and nothing else) to preprocess the dataset without using any depricated libraries.:

        Dataset Information:
        - Columns: {data_analysis['columns']}
        - Data Types: {data_analysis['data_types']}
        - Missing Values: {data_analysis['missing_values']}
        - Target Variable: {target_variable}

        **Transformation Guidelines:**
        1. Use simple, direct preprocessing methods
        2. Handle missing values with basic strategy
        3. Perform minimal feature engineering
        4. Return a clean DataFrame

        Create a function `transform_data(df)` that:
        - Drops columns with too many missing values (>50%)
        - Fills missing numerical values with median
        - Fills missing categorical values with mode
        - Converts categorical variables to numerical using label encoding
        - Ensures all features are numeric
        - Keeps only relevant columns for modeling
        """

        # Use AutoGen to generate transformation code
        code_response = self.user_proxy.initiate_chat(
            self.assistant, 
            message=transformation_request
        )

        # Extract code block
        generated_code = None
        for message in reversed(self.user_proxy.chat_messages[self.assistant]):
            if '```python' in message['content']:
                # More lenient code extraction
                generated_code = message['content'].split('```python')[1].split('```')[0].strip()
                break

        
        generated_code += """
def transform_data1(df):
    # Drop columns with more than 50% missing values
    df = df.dropna(axis=1, thresh=len(df) * 0.5)
    
    # Fill missing numerical values with median
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Fill missing categorical values with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # Convert categorical variables to numerical
    df = pd.get_dummies(df, drop_first=True)
    
    return df
    """
        
        return generated_code

    def execute_transformation_code(self, csv_path: str, transformation_code: str) -> pd.DataFrame:
        # Ensure csv_path is an absolute path within our project structure
        csv_path = self.resolve_path(csv_path)
        
        # Prepare the transformation code with absolute path handling and necessary imports
        exec_code = f"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

{transformation_code}

# Read the CSV using the resolved path
input_df = pd.read_csv(r'{csv_path}')
transformed_df = transform_data(input_df)
transformed_df = transform_data1(transformed_df)
# Save the transformed DataFrame
transformed_path = os.path.join(r'{self.transformed_dir}', 'transformed_data.csv')
transformed_df.to_csv(transformed_path, index=False)
    """

        # Create a local namespace for code execution
        local_namespace = {}
        print("Executing Transformation Code...", exec_code)
        self.transformation = exec_code
        # Execute the code
        exec(exec_code, globals(), local_namespace)
        
        # Read and return the transformed DataFrame
        transformed_path = os.path.join(self.transformed_dir, 'transformed_data.csv')
        return pd.read_csv(transformed_path)


    def prepare_data(self, csv_path: str, target_variable: str):
        """
        Prepare data for machine learning using simplified transformations
        
        Args:
            csv_path (str): Path to the CSV file
            target_variable (str): Target variable for prediction
        
        Returns:
            tuple: Prepared X_train, X_test, y_train, y_test
        """
        # Analyze data
        data_analysis = self.analyze_data(csv_path)
        
        # Generate transformation code
        transformation_code = self.generate_transformation_code(
            data_analysis, 
            target_variable, 
            csv_path
        )
        # print("Generated Transformation Code:\n", transformation_code)
        
        # Execute transformation code
        transformed_df = self.execute_transformation_code(csv_path, transformation_code)
        
        # Ensure target variable is in the transformed DataFrame
        if target_variable not in transformed_df.columns:
            # If target variable was encoded, try to handle it
            original_df = pd.read_csv(csv_path)
            transformed_df[target_variable] = original_df[target_variable]
        
        # Separate features and target
        X = transformed_df.drop(columns=[target_variable])
        y = transformed_df[target_variable]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test
    
    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test, project_name, problem):
        mlflow.set_experiment(project_name)
        
        problem_type = 'classification' if problem == 'C' else 'regression'
        
        models = {
            'classification': [
                ('Logistic_Regression', LogisticRegression()),
                ('Random_Forest', RandomForestClassifier()),
                ('SVM', SVC(probability=True))
            ],
            'regression': [
                ('Linear_Regression', LinearRegression()),
                ('Random_Forest', RandomForestRegressor()),
                ('SVR', SVR())
            ]
        }
        
        best_models = []
        with open("transformed/transformation_code.py", "w") as f:
            f.write(self.transformation)
        
        for name, model in models[problem_type]:
            with mlflow.start_run(run_name=f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as main_run:
                print(f"Starting run for {name}")
                print(f"Run ID: {main_run.info.run_id}")
                
                # Ensure X_train and X_test are DataFrames
                X_train = pd.DataFrame(X_train)
                X_test = pd.DataFrame(X_test)
                
                # Ensure consistent data types
                X_train = X_train.astype('float64')
                X_test = X_test.astype('float64')
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Log parameters
                mlflow.log_params(model.get_params())
                mlflow.log_artifact("transformed/transformation_code.py")
                
                # Log metrics
                if problem_type == 'classification':
                    accuracy = accuracy_score(y_test, y_pred)
                    mlflow.log_metric('accuracy', accuracy)
                    mlflow.log_text(classification_report(y_test, y_pred), "classification_report.txt")
                    print("classification complete")
                else:
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    mlflow.log_metric('mse', mse)
                    mlflow.log_metric('r2', r2)
                    mlflow.log_metric('mae', mae)
                    print("regression complete")
                
                # Infer the model signature
                signature = infer_signature(X_train, model.predict(X_train))
                
                # Create an example input (as a dictionary of column names and values)
                input_example = X_train.iloc[0].to_dict()
                
                # Log model with signature and input example
                mlflow.sklearn.log_model(
                    model, 
                    f"model_{name}", 
                    signature=signature,
                    input_example=input_example
                )
                
                # Log feature importance if available
                if hasattr(model, 'feature_importances_'):
                    feature_importance = pd.DataFrame({
                        'feature': X_train.columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    mlflow.log_table(feature_importance, "feature_importance.json")
                
                # Add to best models list
                best_models.append((name, model))
            mlflow.end_run()
        return best_models
        
    def run_auto_ml_pipeline(self, csv_path: str, target_variable: str, project_name: str, problem:str):
        """
        Run the complete AutoML pipeline
        
        Args:
            csv_path (str): Path to the CSV file
            target_variable (str): Target variable for prediction
            project_name (str): Name of the MLflow project
        """
         # Resolve the input CSV path
        resolved_csv_path = self.resolve_path(csv_path)
        
        print(f"Input CSV Path: {resolved_csv_path}")
        print(f"Project Base Directory: {self.base_dir}")
        # Analyze data
        data_analysis = self.analyze_data(resolved_csv_path)
        print("Data Analysis:\n", data_analysis)
        
        # Generate transformation suggestions
        # self.generate_transformation_code(data_analysis, target_variable, csv_path)
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(csv_path, target_variable)
        
        # Train and evaluate models
        best_models = self.train_and_evaluate_models(
            X_train, X_test, y_train, y_test, project_name, problem
        )
        print("completed")
        # Open MLflow UI
        # os.system('mlflow ui')