

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# Load results
df = pd.read_csv("results.csv")

# Define prompt types and tasks
prompt_types = ['baseline', 'cot_short', 'cot_long', 'cot_bad']
tasks = ['readmission', 'intervention', 'procedure']

# Initialize a dictionary to hold F1 scores
f1_scores_by_prompt = {prompt: {task: [] for task in tasks} for prompt in prompt_types}
shot_range = sorted(df['num_shots'].unique())

# Compute F1 scores grouped by prompt type and num_shots
for prompt in prompt_types:
    for num_shots in shot_range:
        subset = df[(df['prompt_type'] == prompt) & (df['num_shots'] == num_shots)]
        for task in tasks:
            y_true = subset[f'ground_truth_{task}']
            y_pred = subset[f'predicted_label_{task}']
            if y_true.sum() == 0 and y_pred.sum() == 0:
                f1 = 1.0  # Perfect match on all zeros
            else:
                f1 = f1_score(y_true, y_pred, zero_division=0)
            f1_scores_by_prompt[prompt][task].append(f1)

# Plot F1 score curves
for task in tasks:
    plt.figure()
    for prompt in prompt_types:
        plt.plot(shot_range, f1_scores_by_prompt[prompt][task], label=prompt)
    plt.title(f"F1 Score vs Shots for {task.capitalize()}")
    plt.xlabel("Number of Shots")
    plt.ylabel("F1 Score")
    plt.legend(title="Prompt Type")
    plt.grid(True)
    plt.savefig(f"figures/f1_by_shots_{task}.png")
    plt.close()