import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from workflow import AutoMLPipeline
import mlflow
from config import config_list
import tempfile
import os

def main():
    st.title("AutoML Workflow")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
            
        original_df = pd.read_csv(tmp_file_path)
        st.write("Original Data Preview:")
        st.dataframe(original_df.head())
        
        # AutoML parameters
        st.sidebar.header("AutoML Parameters")
        target_variable = st.sidebar.selectbox("Select Target Variable", original_df.columns)
        project_name = st.sidebar.text_input("Project Name", "AutoML Project")
        problem_type = st.sidebar.radio("Problem Type", ["Regression", "Classification"])
        
        if st.sidebar.button("Run AutoML"):
            with st.spinner("Running AutoML pipeline..."):
                automl = AutoMLPipeline(config_list)
                automl.run_auto_ml_pipeline(
                    csv_path=tmp_file_path,
                    target_variable=target_variable,
                    project_name=project_name,
                    problem="R" if problem_type == "Regression" else "C"
                )
                
                # Load transformed data
                transformed_path = os.path.join("transformed", "transformed_data.csv")
                if os.path.exists(transformed_path):
                    transformed_df = pd.read_csv(transformed_path)
                    
                    # Side-by-side comparison
                    st.subheader("Data Comparison")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Original Data")
                        st.dataframe(original_df.head())
                        generate_insights(original_df, "Original")
                    
                    with col2:
                        st.write("Transformed Data")
                        st.dataframe(transformed_df.head())
                        generate_insights(transformed_df, "Transformed")
                    
                    # Add download button for transformed data
                    csv = transformed_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Transformed Data",
                        data=csv,
                        file_name='transformed_data.csv',
                        mime='text/csv'
                    )
                else:
                    st.warning("Transformed data not found")

            st.success("AutoML pipeline completed!")
            
            # MLflow section
            st.subheader("Model Tracking")
            if st.button("Open MLflow UI"):
                os.system('mlflow ui')
                st.markdown("MLflow UI is now running. [Open MLflow UI](http://localhost:5000)")
               

def generate_insights(df, title):
    st.write(f"{title} Data Insights:")
    
    # Basic statistics
    st.write("Basic Statistics:")
    numeric_cols = df.select_dtypes(include=['number']).columns
    st.write(df[numeric_cols].describe())
    
    # Correlation heatmap
    if len(numeric_cols) > 1:
        st.write("Correlation Heatmap:")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.write("Not enough numeric columns for correlation heatmap")
    
    # Distribution plots
    st.write("Distribution Plots:")
    for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        plt.title(f'Distribution of {col}')
        st.pyplot(fig)

if __name__ == "__main__":
    main()
