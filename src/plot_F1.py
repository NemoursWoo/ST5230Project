import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel

def plot_and_test_grouped_metrics(csv_path="results_grouped_metrics.csv", output_dir="figures"):
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(csv_path)

    # === Plot: F1 Score per Prompt Type per Task ===
    tasks = ["readmission", "intervention", "procedure"]
    prompt_types = ["baseline", "cot_short", "cot_long", "cot_bad"]
    shot_levels = [f"shots{i}" for i in range(6)]

    for task in tasks:
        # Plot F1 scores for prompt types
        f1_cols = [f"{p}_{task}_f1" for p in prompt_types]
        f1_data = df[f1_cols].melt(var_name="type", value_name="f1")
        f1_data["prompt"] = f1_data["type"].str.replace(f"_{task}_f1", "", regex=False)

        plt.figure(figsize=(8, 5))
        sns.boxplot(x="prompt", y="f1", data=f1_data)
        plt.title(f"F1 Score by Prompt Type ({task})")
        plt.ylabel("F1 Score")
        plt.savefig(f"{output_dir}/{task}_prompt_f1_boxplot.png")
        plt.close()

        # Paired t-test (baseline vs cot_short) [REMOVED]
        # t_stat, p_val = ttest_rel(df[f"baseline_{task}_f1"], df[f"cot_short_{task}_f1"])
        # print(f"[{task.upper()}] t-test (baseline vs cot_short): t={t_stat:.4f}, p={p_val:.4f}")

        # Plot F1 scores for shot levels
        f1_cols_shots = [f"{s}_{task}_f1" for s in shot_levels]
        f1_data_shots = df[f1_cols_shots].melt(var_name="type", value_name="f1")
        f1_data_shots["shots"] = f1_data_shots["type"].str.extract(r"shots(\d)")

        plt.figure(figsize=(8, 5))
        sns.boxplot(x="shots", y="f1", data=f1_data_shots)
        plt.title(f"F1 Score by Number of Shots ({task})")
        plt.ylabel("F1 Score")
        plt.savefig(f"{output_dir}/{task}_shots_f1_boxplot.png")
        plt.close()

def plot_f1_by_shots_with_prompt_legend(csv_path="results_grouped_metrics.csv", output_dir="figures"):
    import os
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    tasks = ["readmission", "intervention", "procedure"]
    prompt_types = ["baseline", "cot_short", "cot_long", "cot_bad"]
    shot_levels = [f"shots{i}" for i in range(6)]

    for task in tasks:
        plt.figure(figsize=(8, 5))
        for prompt in prompt_types:
            f1_values = []
            for shot in range(6):
                col = f"{prompt}_{task}_f1"
                if col in df.columns:
                    f1_values.append(df[col].mean())
                else:
                    f1_values.append(None)
            plt.plot(range(6), f1_values, marker='o', label=prompt)

        plt.title(f"{task.capitalize()} F1 Score vs. Shots")
        plt.xlabel("Number of Shots")
        plt.ylabel("F1 Score")
        plt.legend(title="Prompt Type")
        plt.xticks(range(len(shot_levels)), shot_levels)
        plt.savefig(f"{output_dir}/{task}_f1_by_shots_prompt.png")
        plt.close()

def plot_f1_boxplots_from_raw(csv_path="results.csv", output_dir="figures"):
    import os
    from sklearn.metrics import f1_score
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    tasks = ["readmission", "intervention", "procedure"]
    prompt_types = df["prompt_type"].unique()
    
    for task in tasks:
        records = []
        for prompt in prompt_types:
            prompt_df = df[df["prompt_type"] == prompt]
            for shot in sorted(prompt_df["num_shots"].unique()):
                shot_df = prompt_df[prompt_df["num_shots"] == shot]
                if len(shot_df) == 0:
                    continue
                y_true = shot_df[f"ground_truth_{task}"]
                y_pred = shot_df[f"predicted_label_{task}"]
                f1 = f1_score(y_true, y_pred, zero_division=0)
                records.append({
                    "prompt_type": prompt,
                    "num_shots": shot,
                    "f1": f1,
                    "task": task
                })
        result_df = pd.DataFrame(records)
        plt.figure(figsize=(8, 5))
        sns.boxplot(x="prompt_type", y="f1", data=result_df[result_df["task"] == task])
        plt.title(f"F1 Score by Prompt Type ({task})")
        plt.ylabel("F1 Score")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{task}_prompt_f1_boxplot_from_raw.png")
        plt.close()
        
        # Plot F1 Score by Shots for current task
        plt.figure(figsize=(8, 5))
        sns.boxplot(x="num_shots", y="f1", data=result_df[result_df["task"] == task])
        plt.title(f"F1 Score by Number of Shots ({task})")
        plt.ylabel("F1 Score")
        plt.xlabel("Number of Shots")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{task}_shots_f1_boxplot_from_raw.png")
        plt.close()

def f1_significance_test_and_heatmap(csv_path="results.csv", output_dir="figures"):
    import os
    os.makedirs(output_dir, exist_ok=True)
    from scipy.stats import ttest_rel
    import seaborn as sns
    from sklearn.metrics import f1_score

    df = pd.read_csv(csv_path)
    tasks = ["readmission", "intervention", "procedure"]
    prompt_types = ["baseline", "cot_short", "cot_long", "cot_bad"]
    prompt_pairs = [("baseline", "cot_short"), ("cot_short", "cot_long"), ("cot_long", "cot_bad")]

    all_results = []

    for task in tasks:
        for shot in sorted(df["num_shots"].unique()):
            df_shot = df[df["num_shots"] == shot]
            for p1, p2 in prompt_pairs:
                f1s_1, f1s_2 = [], []
                for stay_id in df_shot["stay_id"].unique():
                    sub_df = df_shot[df_shot["stay_id"] == stay_id]
                    gt_1 = sub_df[sub_df["prompt_type"] == p1][f"ground_truth_{task}"]
                    pred_1 = sub_df[sub_df["prompt_type"] == p1][f"predicted_label_{task}"]
                    gt_2 = sub_df[sub_df["prompt_type"] == p2][f"ground_truth_{task}"]
                    pred_2 = sub_df[sub_df["prompt_type"] == p2][f"predicted_label_{task}"]
                    if len(gt_1) > 0 and len(gt_2) > 0:
                        f1_1 = f1_score(gt_1, pred_1, zero_division=0)
                        f1_2 = f1_score(gt_2, pred_2, zero_division=0)
                        f1s_1.append(f1_1)
                        f1s_2.append(f1_2)
                if len(f1s_1) == len(f1s_2) and len(f1s_1) > 0:
                    t_stat, p_val = ttest_rel(f1s_1, f1s_2)
                    all_results.append({
                        "task": task,
                        "num_shots": shot,
                        "comparison": f"{p1} vs {p2}",
                        "t_stat": t_stat,
                        "p_value": p_val
                    })

    result_df = pd.DataFrame(all_results)
    result_df.to_csv(f"{output_dir}/f1_significance_tests.csv", index=False)
    print(f"Saved significance test results to {output_dir}/f1_significance_tests.csv")

    for shot in sorted(df["num_shots"].unique()):
        df_shot = df[df["num_shots"] == shot]
        heat_data = []
        for p in prompt_types:
            p_df = df_shot[df_shot["prompt_type"] == p]
            row = []
            for task in tasks:
                f1 = f1_score(p_df[f"ground_truth_{task}"], p_df[f"predicted_label_{task}"], zero_division=0)
                row.append(f1)
            heat_data.append(row)
        heat_df = pd.DataFrame(heat_data, columns=tasks, index=prompt_types)
        plt.figure(figsize=(8, 5))
        sns.heatmap(heat_df, annot=True, fmt=".2f", cmap="YlGnBu")
        plt.title(f"F1 Score Heatmap by Prompt Type (Shot {shot})")
        plt.savefig(f"{output_dir}/f1_heatmap_shot{shot}.png")
        plt.close()

if __name__ == "__main__":
    # plot_and_test_grouped_metrics()
    # plot_f1_by_shots_with_prompt_legend()
    plot_f1_boxplots_from_raw()
    # f1_significance_test_and_heatmap()