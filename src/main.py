import os
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import torch

from openai import OpenAI
from .gpt4_api import call_gpt  # Importing the GPT API call function
from .prompt_templates import prompt_types  # Importing prompt templates and few-shot examples

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------------
# Set OpenAI API key from environment variable or directly assign if needed

# Define a simple text cleaning function: lowercase, remove punctuation, and extra spaces.
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    return text

# Define an auxiliary function to generate a simple clinical summary in English.
def create_clinical_summary(row):
    summary = f"Patient ID {row['subject_id']}, ICU length-of-stay {row['los']:.1f} days, average heart rate {row['avg_heart_rate']:.1f} bpm."
    return summary

# Data Loading & Preprocessing
def load_or_generate_icu_summary():
    # Check if the CSV file exists
    if os.path.exists("data/icu_summary.csv"):
        return pd.read_csv("data/icu_summary.csv")
    
    # 1.1 Read CSV files (modify the file paths as needed)
    caregiver = pd.read_csv('scratch/caregiver.csv.gz', compression='gzip')
    d_items = pd.read_csv('scratch/d_items.csv.gz', compression='gzip')
    chartevents = pd.read_csv('scratch/chartevents.csv.gz', low_memory=False, compression='gzip')
    icustays = pd.read_csv('scratch/icustays.csv.gz', compression='gzip')
    inputevents = pd.read_csv('scratch/inputevents.csv.gz', low_memory=False, compression='gzip')

    # Initial Exploration
    print("Preview of caregiver table:")
    print(caregiver.head())
    print("\nInfo for d_items table:")
    print(d_items.info())
    print("\nFirst 5 rows of chartevents:")
    print(chartevents.head())
    print("\nFirst 5 rows of icustays:")
    print(icustays.head())

    # 1.2 Data Cleaning
    # Remove duplicate records
    chartevents.drop_duplicates(inplace=True)
    icustays.drop_duplicates(inplace=True)

    # Drop rows with missing values in key numerical field 'valuenum' in chartevents
    chartevents_clean = chartevents.dropna(subset=['valuenum'])

    # 1.3 Convert time columns to datetime (for chartevents and icustays)
    chartevents_clean['charttime'] = pd.to_datetime(chartevents_clean['charttime'])
    chartevents_clean['storetime'] = pd.to_datetime(chartevents_clean['storetime'])
    icustays['intime'] = pd.to_datetime(icustays['intime'])
    icustays['outtime'] = pd.to_datetime(icustays['outtime'])

    # 1.4 Merge chartevents with d_items to add measurement descriptions.
    # Use a subset of fields: itemid, label, unitname, lownormalvalue, highnormalvalue.
    d_items_subset = d_items[['itemid', 'label', 'unitname', 'lownormalvalue', 'highnormalvalue']]
    chartevents_merged = pd.merge(chartevents_clean, d_items_subset, on='itemid', how='left')

    # 1.5 Filter heart rate records.
    # Assume that heart rate records have label "Heart Rate" (adjust if needed).
    heart_rate_events = chartevents_merged[chartevents_merged['label'] == 'Heart Rate']
    heart_rate_events['valuenum'] = pd.to_numeric(heart_rate_events['valuenum'], errors='coerce')
    heart_rate_events = heart_rate_events.dropna(subset=['valuenum'])

    # 1.6 Calculate average heart rate per ICU stay (using 'stay_id')
    heart_rate_grouped = heart_rate_events.groupby('stay_id')['valuenum'].mean().reset_index()
    heart_rate_grouped.rename(columns={'valuenum': 'avg_heart_rate'}, inplace=True)
    print("\nSample of average heart rate per ICU stay:")
    print(heart_rate_grouped.head())

    # 1.7 Merge average heart rate with icustays data.
    icu_merged = pd.merge(icustays, heart_rate_grouped, on='stay_id', how='left')
    icu_merged = icustays.merge(heart_rate_grouped, on='stay_id', how='left')
    icu_merged = icu_merged.dropna(subset=['avg_heart_rate'])
    print("\nMerged ICU stay records:")
    print(icu_merged[['stay_id', 'subject_id', 'intime', 'outtime', 'los', 'avg_heart_rate']].head())

    # 1.8 Standardize avg_heart_rate (optional, can be used for modeling)
    scaler = StandardScaler()
    icu_merged['avg_heart_rate_scaled'] = scaler.fit_transform(icu_merged[['avg_heart_rate']])

    # 1.9 Textual Data Processing
    # Example: Clean the "ordercategorydescription" field from inputevents (if available).
    if 'ordercategorydescription' in inputevents.columns:
        inputevents['cleaned_ordercat'] = inputevents['ordercategorydescription'].apply(clean_text)
        print("\nSample cleaned order category descriptions from inputevents:")
        print(inputevents['cleaned_ordercat'].unique()[:5])

    # Demonstrate TF-IDF vectorization on the cleaned text (optional, just for feature extraction demonstration).
    if 'cleaned_ordercat' in inputevents.columns:
        vectorizer = TfidfVectorizer(max_features=100)
        ordercat_corpus = inputevents['cleaned_ordercat'].dropna().tolist()
        tfidf_matrix = vectorizer.fit_transform(ordercat_corpus)
        print("\nSample TF-IDF keywords:")
        print(vectorizer.get_feature_names_out()[:10])

    # 1.10 Construct Clinical Summaries
    # Generate a clinical summary for each ICU stay.
    icu_merged['clinical_summary'] = icu_merged.apply(create_clinical_summary, axis=1)

    # Save the processed DataFrame to CSV
    icu_merged.to_csv("data/icu_summary.csv", index=False)
    return icu_merged

def run_icl_experiment(icu_merged):
    # For demonstration, randomly sample 3 ICU records and obtain their clinical summaries.
    sampled_icu = icu_merged.sample(n=3, random_state=42).copy()

    # Initialize lists to store prediction results.
    baseline_results = []
    cot_results = []

    print("====== Baseline ICL Experiment ======")
    for idx, row in sampled_icu.iterrows():
        test_input = row['clinical_summary']
        prompt_baseline = prompt_types["baseline"](test_input)  # Using the imported prompt template
        print("Generated prompt (Baseline):")
        print(prompt_baseline)
        prediction_text = call_gpt(prompt_baseline, max_tokens=100)  # Using the imported GPT API call
        # Extract the prediction from the returned text (assumes the answer contains "Prediction:" followed by the result).
        if "Prediction:" in prediction_text:
            predicted_label = prediction_text.split("Prediction:")[-1].strip().split("\n")[0]
        else:
            predicted_label = prediction_text.strip().split("\n")[0]
        baseline_results.append(predicted_label)
        print("Baseline Prediction:", predicted_label)
        print("--------------------------------------------------\n")

    print("====== CoT-enhanced ICL Experiment ======")
    for idx, row in sampled_icu.iterrows():
        test_input = row['clinical_summary']
        prompt_cot = prompt_types["cot_short"](test_input)  # Using the imported prompt template
        print("Generated prompt (CoT):")
        print(prompt_cot)
        prediction_text = call_gpt(prompt_cot, max_tokens=150)  # Using the imported GPT API call
        if "Prediction:" in prediction_text:
            predicted_label = prediction_text.split("Prediction:")[-1].strip().split("\n")[0]
        else:
            predicted_label = prediction_text.strip().split("\n")[0]
        cot_results.append(predicted_label)
        print("CoT Prediction:", predicted_label)
        print("--------------------------------------------------\n")

    # Save test samples and prediction results to a CSV for further analysis.
    results_df = sampled_icu[['stay_id', 'subject_id', 'clinical_summary']].copy()
    results_df['Baseline_Prediction'] = baseline_results
    results_df['CoT_Prediction'] = cot_results
    results_df.to_csv('icu_gpt4_cot_results.csv', index=False)
    print("Experimental results have been saved to icu_gpt4_cot_results.csv")

def run_cot_analysis(sampled_icu):
    # Implement consistency and rationality assessment of CoT outputs here.
    # Placeholder for actual analysis logic.
    analysis_results = {
        'stay_id': sampled_icu['stay_id'],
        'subject_id': sampled_icu['subject_id'],
        'CoT_Prediction': sampled_icu['CoT_Prediction'],
        'Consistency_Score': [1.0] * len(sampled_icu),  # Dummy consistency scores
        'Rationality_Score': [1.0] * len(sampled_icu),  # Dummy rationality scores
    }
    analysis_df = pd.DataFrame(analysis_results)
    analysis_df.to_csv('cot_analysis_results.csv', index=False)
    print("CoT analysis results have been saved to cot_analysis_results.csv")

# Load or generate the ICU summary data
icu_merged = load_or_generate_icu_summary()

# Call the ICL experiment and CoT analysis functions
run_icl_experiment(icu_merged)
sampled_icu = icu_merged.sample(n=3, random_state=42).copy()
run_cot_analysis(sampled_icu)