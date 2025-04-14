import re

baseline_prompt = lambda input_text: f"Input: {input_text}\nPlease answer strictly in the following format:\n1. Reasoning: [reply with your explanation here]\n2. Prediction: [only reply with one of: low risk, high risk, uncertain]\n"

def cot_short_prompt(input_text):
    # Extract los and hr from input_text
    los_match = re.search(r"ICU length-of-stay ([\d\.]+)", input_text)
    hr_match = re.search(r"average heart rate ([\d\.]+)", input_text)
    los = float(los_match.group(1)) if los_match else 0
    hr = float(hr_match.group(1)) if hr_match else 0

    # Construct reasoning based on extracted values
    prompt = f"The patient has an ICU stay of {los} days and an average heart rate of {hr} bpm."
    
    return f"Input: {input_text}\nPrompt: {prompt}\nPlease answer strictly in the following format:\n1. Reasoning: [reply with your explanation here]\n2. Prediction: [only reply with one of: low risk, high risk, uncertain]\n"

def cot_long_prompt(input_text):
    # Extract los and hr from input_text
    los_match = re.search(r"ICU length-of-stay ([\d\.]+)", input_text)
    hr_match = re.search(r"average heart rate ([\d\.]+)", input_text)
    los = float(los_match.group(1)) if los_match else 0
    hr = float(hr_match.group(1)) if hr_match else 0

    # Construct reasoning based on extracted values
    prompt = f"Let's think step by step. Normally, if a patient has an ICU stay over 4 days or an average heart rate over 10 days, we consider him/her at a high risk. This patient has an ICU stay of {los} days and an average heart rate of {hr} bpm."
    
    return f"Input: {input_text}\nPrompt: {prompt}\nPlease answer strictly in the following format:\n1. Reasoning: [reply with your explanation here]\n2. Prediction: [only reply with one of: low risk, high risk, uncertain]\n"

def cot_bad_prompt(input_text):
    # Extract los and hr from input_text
    los_match = re.search(r"ICU length-of-stay ([\d\.]+)", input_text)
    hr_match = re.search(r"average heart rate ([\d\.]+)", input_text)
    los = float(los_match.group(1)) if los_match else 0
    hr = float(hr_match.group(1)) if hr_match else 0

    # Construct reasoning based on extracted values
    prompt = f"Let's think step by step. Normally, if a patient has an ICU stay over 4 days or an average heart rate over 10 days, we consider him/her at a low risk. This patient has an ICU stay of {los} days and an average heart rate of {hr} bpm."
    
    return f"Input: {input_text}\nPrompt: {prompt}\nPlease answer strictly in the following format:\n1. Reasoning: [reply with your explanation here]\n2. Prediction: [only reply with one of: low risk, high risk, uncertain]\n"

prompt_types = {
    "baseline": baseline_prompt,
    "cot_short": cot_short_prompt,
    "cot_long": cot_long_prompt,
    "cot_bad": cot_bad_prompt,
}