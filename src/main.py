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

def label_readmission(icu_df):
    readmit_flags = icu_df.groupby("subject_id")['stay_id'].transform('count') > 1
    return readmit_flags.astype(int)

def label_critical_intervention(inputevents, d_items):
    critical_keywords = ["norepinephrine", "vasopressin", "epinephrine", "dopamine"]
    critical_ids = d_items[d_items['label'].str.lower().isin(critical_keywords)]['itemid'].tolist()
    critical_input = inputevents[inputevents['itemid'].isin(critical_ids)]
    critical_flags = critical_input[['subject_id', 'stay_id']].drop_duplicates()
    critical_flags['critical_intervention'] = 1
    return critical_flags

def label_procedures(procedureevents, d_items):
    keywords = ["ventilation", "dialysis"]
    proc_ids = d_items[d_items['label'].str.lower().str.contains('|'.join(keywords))]['itemid'].tolist()
    proc_records = procedureevents[procedureevents['itemid'].isin(proc_ids)]
    proc_flags = proc_records[['subject_id', 'stay_id']].drop_duplicates()
    proc_flags['procedures'] = 1
    return proc_flags

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

    # Load additional files
    procedureevents = pd.read_csv('scratch/procedureevents.csv.gz', compression='gzip')

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

    # Generate readmission labels
    icu_merged['readmission'] = label_readmission(icu_merged)

    # Generate critical intervention labels
    critical_df = label_critical_intervention(inputevents, d_items)
    icu_merged = pd.merge(icu_merged, critical_df, on=['subject_id', 'stay_id'], how='left')
    icu_merged['critical_intervention'] = icu_merged['critical_intervention'].fillna(0).astype(int)

    # Generate procedure-related labels
    proc_df = label_procedures(procedureevents, d_items)
    icu_merged = pd.merge(icu_merged, proc_df, on=['subject_id', 'stay_id'], how='left')
    icu_merged['procedures'] = icu_merged['procedures'].fillna(0).astype(int)

    # Save the processed DataFrame to CSV
    icu_merged.to_csv("data/icu_summary.csv", index=False)
    return icu_merged


def run_icl_experiment(icu_merged, n=1, max_num_shots=5):
    # For demonstration, randomly sample max_num_shots + n ICU records and obtain their clinical summaries.
    sampled_icu = icu_merged.sample(n=max_num_shots + n, random_state=42).copy()
    
    # Split into test_set and support_set
    test_set = sampled_icu.iloc[:n]
    support_set = sampled_icu.iloc[n:max_num_shots + n]

    # Initialize lists to store prediction results and reasoning.
    results = []

    few_shot_prompts = []
    for _, row in support_set.iterrows():
        los_match = re.search(r"ICU length-of-stay ([\d\.]+)", row['clinical_summary'])
        hr_match = re.search(r"average heart rate ([\d\.]+)", row['clinical_summary'])
        los = float(los_match.group(1)) if los_match else 0
        hr = float(hr_match.group(1)) if hr_match else 0
        prompt = f"Input: {row['clinical_summary']}\nPrompt: The patient has an ICU stay of {los} days and an average heart rate of {hr} bpm.\nPrediction: Readmission:{row['readmission']}\n Critical Intervention:{row['critical_intervention']}\n Ventilation or Dialysis Procedures:{row['procedures']}\n"
        few_shot_prompts.append(prompt)

    prompt_types_list = ["baseline", "cot_short", "cot_long", "cot_bad"]

    print("====== ICL Experiment ======")
    for idx, row in test_set.iterrows():
        test_input = row['clinical_summary']
        ground_truth_readmission = row['readmission']
        ground_truth_intervention = row['critical_intervention']
        ground_truth_procedure = row['procedures']
        
        for num_shots in range(max_num_shots + 1):
            few_shot_prompt = '\n'.join(few_shot_prompts[:num_shots]) + '\n'
            
            for prompt_type in prompt_types_list:
                prompt = prompt_types[prompt_type](test_input)  # Using the imported prompt template
                full_prompt = f"{few_shot_prompt}{prompt}"
                print(f"Generated prompt ({prompt_type}):")
                print(full_prompt)
                prediction_text = call_gpt(full_prompt, max_tokens=1000)  # Using the imported GPT API call
                if "Readmission:" in prediction_text:
                    predicted_label_readmission = prediction_text.split("Readmission:")[-1].strip().split("\n")[0]
                else:
                    predicted_label_readmission = prediction_text.strip().split("\n")[0]
                if "Critical Intervention:" in prediction_text:
                    predicted_label_intervention = prediction_text.split("Critical Intervention:")[-1].strip().split("\n")[0]
                else:
                    predicted_label_intervention = prediction_text.strip().split("\n")[0]
                if "Ventilation or Dialysis Procedures:" in prediction_text:
                    predicted_label_procedure = prediction_text.split("Ventilation or Dialysis Procedures:")[-1].strip().split("\n")[0]
                else:
                    predicted_label_procedure = prediction_text.strip().split("\n")[0]
                predicted_label_readmission = int(predicted_label_readmission)
                predicted_label_intervention = int(predicted_label_intervention)
                predicted_label_procedure = int(predicted_label_procedure)
                reasoning = ""
                if "Reasoning:" in prediction_text:
                    reasoning = prediction_text.split("Reasoning:")[-1].strip().split("\n")[0]
                accuracy_readmission = int(predicted_label_readmission == ground_truth_readmission)  # Calculate accuracy
                accuracy_intervention = int(predicted_label_intervention == ground_truth_intervention)  # Calculate accuracy
                accuracy_procedure = int(predicted_label_procedure == ground_truth_procedure)  # Calculate accuracy
                results.append({
                    'stay_id': row['stay_id'],
                    'subject_id': row['subject_id'],
                    'clinical_summary': row['clinical_summary'],
                    'ground_truth_readmission': ground_truth_readmission,
                    'ground_truth_intervention': ground_truth_intervention,
                    'ground_truth_procedure': ground_truth_procedure,
                    'num_shots': num_shots,
                    'predicted_label_readmission': predicted_label_readmission,
                    'predicted_label_intervention': predicted_label_intervention,
                    'predicted_label_procedure': predicted_label_procedure,
                    'reasoning': reasoning,
                    'accuracy_readmission': accuracy_readmission,
                    'accuracy_intervention': accuracy_intervention,
                    'accuracy_procedure': accuracy_procedure,
                    'prompt_type': prompt_type
                })
                print(f"{prompt_type} Prediction:", predicted_label_readmission, predicted_label_intervention, predicted_label_procedure)
                print("--------------------------------------------------\n")

    # Create a DataFrame to return with predictions and evaluations
    results_df = pd.DataFrame(results)
    return results_df

def run_cot_analysis(sampled_df):
    # Implement consistency and rationality assessment of CoT outputs here.
    # Score reasoning using GPT
    scores = []
    for reasoning in sampled_df['reasoning']:
        score = call_gpt(f"Please evaluate the following clinical reasoning. Give **ONLY** numeric scores from 1 to 5 for each of the following: 1. Clarity: 2. Relevance: 3. Medical Soundness: \n\nReasoning: {reasoning}")  # Updated prompt for structured scoring
        print(score)
        clarity = re.search(r'Clarity:\s*(\d)', score)
        relevance = re.search(r'Relevance:\s*(\d)', score)
        medical_soundness = re.search(r'Medical Soundness:\s*(\d)', score)
        clarity_score = int(clarity.group(1)) if clarity else None
        relevance_score = int(relevance.group(1)) if relevance else None
        medical_soundness_score = int(medical_soundness.group(1)) if medical_soundness else None
        scores.append((clarity_score, relevance_score, medical_soundness_score))

    # Add scores to DataFrame
    sampled_df[['Clarity_Score', 'Relevance_Score', 'Medical_Soundness_Score']] = pd.DataFrame(scores, index=sampled_df.index)
    sampled_df.to_csv('cot_analysis_results.csv', index=False)
    print("CoT analysis results have been saved to cot_analysis_results.csv")

# Load or generate the ICU summary data
icu_merged = load_or_generate_icu_summary()

# Call the ICL experiment and CoT analysis functions
sampled_df = run_icl_experiment(icu_merged)
run_cot_analysis(sampled_df)