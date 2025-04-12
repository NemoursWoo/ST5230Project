baseline_prompt = lambda input_text: f"Input: {input_text}\nPrediction:"

cot_short_prompt = lambda input_text: f"Input: {input_text}\nReasoning: The patient has a relatively short ICU stay and a normal heart rate, suggesting low risk.\nPrediction:"

cot_long_prompt = lambda input_text: f"Input: {input_text}\nReasoning: Let's analyze this step by step. The length of stay in the ICU is relatively short, which usually indicates a more stable condition. The heart rate is within the normal range (60-100 bpm), suggesting cardiovascular stability. Combining these two factors, the patient is likely at low risk.\nPrediction:"

cot_bad_prompt = lambda input_text: f"Input: {input_text}\nReasoning: The patient has a short ICU stay and a normal heart rate, therefore the patient is probably at high risk because everything looks okay.\nPrediction:"

prompt_types = {
    "baseline": baseline_prompt,
    "cot_short": cot_short_prompt,
    "cot_long": cot_long_prompt,
    "cot_bad": cot_bad_prompt,
}