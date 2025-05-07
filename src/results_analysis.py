import pandas as pd
from scipy.stats import ttest_rel
import seaborn as sns
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("results.csv")

# === 1. Significance Testing between Prompt Types ===
def run_significance_testing(df):
    tasks = ['f1_readmission', 'f1_intervention', 'f1_procedure']
    prompt_pairs = [('baseline', 'cot_short'), ('cot_short', 'cot_long'), ('cot_long', 'cot_bad')]
    all_results = []

    for shot in sorted(df["num_shots"].unique()):
        df_shot = df[df["num_shots"] == shot]
        shot_results = []
        for task in tasks:
            for p1, p2 in prompt_pairs:
                group1 = df_shot[df_shot["prompt_type"] == p1][task]
                group2 = df_shot[df_shot["prompt_type"] == p2][task]
                if len(group1) == len(group2) and len(group1) > 0:
                    t_stat, p_value = ttest_rel(group1, group2)
                    shot_results.append({
                        'num_shots': shot,
                        'task': task,
                        'prompt_pair': f"{p1} vs {p2}",
                        't_stat': t_stat,
                        'p_value': p_value
                    })
        significance_df = pd.DataFrame(shot_results)
        significance_df.to_csv(f"significance_testing_f1_shot{shot}.csv", index=False)
        all_results.extend(shot_results)

    print("Saved significance test results by shot to significance_testing_shot*.csv")

# === 2. Cross-task Generalization Analysis ===
def plot_cross_task_heatmap(df):
    for shot in sorted(df["num_shots"].unique()):
        df_shot = df[df["num_shots"] == shot]
        task_f1s = df_shot.groupby("prompt_type")[
            ["f1_readmission", "f1_intervention", "f1_procedure"]
        ].mean()

        plt.figure(figsize=(8, 5))
        sns.heatmap(task_accuracies, annot=True, cmap="YlGnBu", fmt=".2f")
        plt.title(f"Shot {shot}: Mean F1 per Prompt Type Across Tasks")
        plt.savefig(f"figures/cross_task_f1_heatmap_shot{shot}.png")
        plt.close()
        print(f"Saved cross-task generalization heatmap for shot {shot} to figures/cross_task_accuracy_heatmap_shot{shot}.png")

if __name__ == "__main__":
    run_significance_testing(df)
    plot_cross_task_heatmap(df)