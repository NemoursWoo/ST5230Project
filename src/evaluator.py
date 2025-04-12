from gpt4_api import call_gpt

def evaluate_reasoning_quality(reasoning_text):
    prompt = f"Please read the following medical reasoning and rate its logical quality from 1 (very poor) to 5 (very good).\nReasoning: "{reasoning_text}"\nRating (1-5):"
    response = call_gpt(prompt, max_tokens=10, temperature=0)
    return response