import pandas as pd
import random
from prompt_templates import prompt_types
from gpt4_api import call_gpt
from evaluator import evaluate_reasoning_quality

df = pd.read_csv("data/icu_summary.csv")
sampled = df.sample(n=5, random_state=42)

results = []

for idx, row in sampled.iterrows():
    input_text = row["clinical_summary"]

    for prompt_type, prompt_func in prompt_types.items():
        prediction_list = []
        reasoning_extracted = ""

        for _ in range(3):  # Repeat to test consistency
            prompt = prompt_func(input_text)
            reply = call_gpt(prompt, max_tokens=100, temperature=0.7)
            prediction = "low risk" if "low" in reply.lower() else "high risk"
            prediction_list.append(prediction)

            if "reasoning:" in reply.lower():
                reasoning_extracted = reply.lower().split("reasoning:")[-1].strip().split("prediction:")[0].strip()

        is_consistent = len(set(prediction_list)) == 1
        reasoning_score = evaluate_reasoning_quality(reasoning_extracted) if reasoning_extracted else "NA"

        results.append({
            "id": idx,
            "input": input_text,
            "prompt_type": prompt_type,
            "predictions": prediction_list,
            "consistent": is_consistent,
            "reasoning": reasoning_extracted,
            "reasoning_score": reasoning_score
        })

df_result = pd.DataFrame(results)
df_result.to_csv("results/cot_results.csv", index=False)
print("Results saved to results/cot_results.csv")