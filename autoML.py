import os
import argparse
from workflow import AutoMLPipeline
from dotenv import load_dotenv
from config import config_list
def main():
    # Load environment variables
    load_dotenv()

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run AutoML pipeline on a CSV file.')
    parser.add_argument('--path', type=str, default='D:/project/AutoMl', help='Path to the directory containing the CSV file')
    parser.add_argument('--project_name', type=str, required=True, help='Name of the project')
    parser.add_argument('--target_variable', type=str, required=True, help='Name of the target variable')
    parser.add_argument('--problem', type=str, default='R', help='what type of problem you are solving?, R for Regression and C for classification')

    # Parse arguments
    args = parser.parse_args()

    # Initialize AutoMLPipeline
    automl = AutoMLPipeline(config_list)

    # Run the AutoML pipeline
    automl.run_auto_ml_pipeline(
        csv_path=args.path, 
        target_variable=args.target_variable, 
        project_name=args.project_name,
        problem=args.problem
    )

if __name__ == "__main__":
    main()
#python autoML.py --path "D:\project\AutoMl\Student_Mental_Stress_and_Coping_Mechanisms.csv" --project "Student stress" --target_variable "Substance Use"