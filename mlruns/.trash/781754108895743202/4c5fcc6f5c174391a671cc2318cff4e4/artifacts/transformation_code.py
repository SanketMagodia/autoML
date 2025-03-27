
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def transform_data(df):
    # Drop columns with too many missing values (>50%)
    df = df.dropna(thresh=len(df)*0.5, axis=1)

    # Fill missing numerical values with median
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

    # Fill missing categorical values with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode().iloc[0])

    # Convert categorical variables to numerical using label encoding
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    # Ensure all features are numeric
    df = df.select_dtypes(include=['int64', 'float64'])

    # Keep only relevant columns for modeling
    relevant_cols = ['Age', 'Academic Performance (GPA)', 'Study Hours Per Week', 
                      'Social Media Usage (Hours per day)', 'Sleep Duration (Hours per night)', 
                      'Physical Exercise (Hours per week)', 'Family Support  ', 'Financial Stress', 
                      'Peer Pressure', 'Relationship Stress', 'Mental Stress Level', 'Diet Quality', 
                      'Cognitive Distortions', 'Substance Use']
    relevant_cols = [col for col in relevant_cols if col in df.columns]
    df = df[relevant_cols]

    return df

# Read the CSV using the resolved path
input_df = pd.read_csv(r'D:\project\AutoMl\input\tmp7ebg20b1.csv')
transformed_df = transform_data(input_df)

# Save the transformed DataFrame
transformed_path = os.path.join(r'D:\project\AutoMl\transformed', 'transformed_data.csv')
transformed_df.to_csv(transformed_path, index=False)
    