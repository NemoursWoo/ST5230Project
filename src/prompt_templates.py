import re

baseline_prompt = lambda input_text: f"Input: {input_text}\nPlease answer strictly in the following format:\n1. Reasoning: [reply with your explanation here]\n2. Prediction of Readmission: [only reply with 1 or 0]\n3. Prediction of Critical Intervention: [only reply with 1 or 0]\n4. Prediction of Ventilation or Dialysis Procedures: [only reply with 1 or 0]\n"

def cot_short_prompt(input_text):
    # Extract los and hr from input_text
    los_match = re.search(r"ICU length-of-stay ([\d\.]+)", input_text)
    hr_match = re.search(r"average heart rate ([\d\.]+)", input_text)
    los = float(los_match.group(1)) if los_match else 0
    hr = float(hr_match.group(1)) if hr_match else 0

    # Construct reasoning based on extracted values
    prompt = f"The patient has an ICU stay of {los} days and an average heart rate of {hr} bpm. Based on these two indicators, let's briefly reason whether this implies readmission, need for critical intervention, or severe procedures like ventilation or dialysis."
    
    return f"Input: {input_text}\nPrompt: {prompt}\nPlease answer strictly in the following format:\n1. Reasoning: [reply with your explanation here]\n2. Prediction of Readmission: [only reply with 1 or 0]\n3. Prediction of Critical Intervention: [only reply with 1 or 0]\n4. Ventilation or Dialysis Procedures: [only reply with 1 or 0]\n"

def cot_long_prompt(input_text):
    # Extract los and hr from input_text
    los_match = re.search(r"ICU length-of-stay ([\d\.]+)", input_text)
    hr_match = re.search(r"average heart rate ([\d\.]+)", input_text)
    los = float(los_match.group(1)) if los_match else 0
    hr = float(hr_match.group(1)) if hr_match else 0

    # Construct reasoning based on extracted values
    prompt = f"Let's reason step-by-step. ICU stays longer than 4 days may indicate complications or slow recovery. An average heart rate significantly higher than normal (>100 bpm) can also suggest instability. This patient stayed {los} days and had a heart rate of {hr} bpm. Assess whether they are likely to be readmitted, receive critical interventions, or undergo ventilation/dialysis."
    
    return f"Input: {input_text}\nPrompt: {prompt}\nPlease answer strictly in the following format:\n1. Reasoning: [reply with your explanation here]\n2. Prediction of Readmission: [only reply with 1 or 0]\n2. Prediction of Critical Intervention: [only reply with 1 or 0]\n4. Prediction of Ventilation or Dialysis Procedures: [only reply with 1 or 0]\n"

def cot_bad_prompt(input_text):
    # Extract los and hr from input_text
    los_match = re.search(r"ICU length-of-stay ([\d\.]+)", input_text)
    hr_match = re.search(r"average heart rate ([\d\.]+)", input_text)
    los = float(los_match.group(1)) if los_match else 0
    hr = float(hr_match.group(1)) if hr_match else 0

    # Construct reasoning based on extracted values
    prompt = f"Let's reason step-by-step. ICU stays longer than 4 days don't indicate complications or slow recovery. An average heart rate significantly higher than normal (>100 bpm) cannot suggest instability. This patient stayed {los} days and had a heart rate of {hr} bpm. Assess whether they are likely to be readmitted, receive critical interventions, or undergo ventilation/dialysis."
    
    return f"Input: {input_text}\nPrompt: {prompt}\nPlease answer strictly in the following format:\n1. Reasoning: [reply with your explanation here]\n2. Prediction of Readmission: [only reply with 1 or 0]\n3. Prediction of Critical Intervention: [only reply with 1 or 0]\n4. Prediction of Ventilation or Dialysis Procedures: [only reply with 1 or 0]\n"

prompt_types = {
    "baseline": baseline_prompt,
    "cot_short": cot_short_prompt,
    "cot_long": cot_long_prompt,
    "cot_bad": cot_bad_prompt,
}