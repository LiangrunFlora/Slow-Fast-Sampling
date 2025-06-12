import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
# Ensure the ./speed_flops directory exists
output_dir = './speed_flops'
os.makedirs(output_dir, exist_ok=True)

# Read the CSV file
csv_file = os.path.join(output_dir, 'flops_results_acc.csv')
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"CSV file {csv_file} does not exist. Please run the experiment to generate it.")

df = pd.read_csv(csv_file)

# Check if required columns exist
required_columns = ['avg_flops_per_token', 'arg_run_time_per_token', 'acc', 'prompt_interval_steps', 'gen_interval_steps', 'transfer_ratio', 'steps']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns in CSV: {missing_columns}")

# Function to assign legend labels
def get_legend_label(row):
    p_steps = row['prompt_interval_steps']
    g_steps = row['gen_interval_steps']
    t_ratio = row['transfer_ratio']
    steps = row['steps']
    
    if p_steps == -1 and t_ratio == -1:
        if steps == 256:
            return 'baseline in 256 steps'
        elif steps == 128:
            return 'baseline in 128 steps'
        elif steps == 64:
            return 'baseline in 64 steps'
        elif steps == 32:
            return 'baseline in 32 steps'
        elif steps == 16:
            return 'baseline in 16 steps'
        elif steps == 8:
            return 'baseline in 8 steps'
        elif steps == 4:
            return 'baseline in 4 steps'
    return f'g_interval={int(g_steps)}, transfer_ratio={t_ratio}'

# Add legend labels to the DataFrame
df['legend_label'] = df.apply(get_legend_label, axis=1)

# Get unique legend labels and baseline labels
unique_labels = df['legend_label'].unique()
baseline_labels = ['baseline in 256 steps', 'baseline in 128 steps', 'baseline in 64 steps', 
                  'baseline in 32 steps', 'baseline in 16 steps', 'baseline in 8 steps', 'baseline in 4 steps']

# Get the y-value for the horizontal line (baseline in 256 steps accuracy)
baseline_256 = df[df['legend_label'] == 'baseline in 256 steps']
if not baseline_256.empty:
    baseline_256_acc = baseline_256['acc'].iloc[0]
else:
    baseline_256_acc = None

# Get unique prompt_interval_steps values (excluding baseline where prompt_interval_steps = -1)
prompt_intervals = df[df['prompt_interval_steps'] != -1]['prompt_interval_steps'].unique()

# Define a color cycle for different prompt_interval_steps
colors = plt.cm.tab10(np.linspace(0, 1, len(prompt_intervals)))

# Plot 1: FLOPs vs Accuracy
plt.figure(figsize=(10, 6))

# Plot baseline settings with connected lines and individual legend entries
if all(label in df['legend_label'].values for label in baseline_labels):
    baseline_subset = df[df['legend_label'].isin(baseline_labels)]
    baseline_subset = baseline_subset.sort_values(by='steps')
    plt.plot(
        baseline_subset['avg_flops_per_token'],
        baseline_subset['acc'],
        color='blue',
        linewidth=2,
        linestyle='-',
        label='baseline',
        zorder=1
    )
    for label in baseline_labels:
        subset = df[df['legend_label'] == label]
        if not subset.empty:
            plt.scatter(
                subset['avg_flops_per_token'],
                subset['acc'],
                marker='o',
                label=label,
                s=36,
                zorder=2
            )

# Plot non-baseline settings with lines connecting same prompt_interval_steps
for i, p_interval in enumerate(prompt_intervals):
    subset = df[df['prompt_interval_steps'] == p_interval]
    subset = subset.sort_values(by='avg_flops_per_token')  # Sort to ensure proper line connection
    plt.plot(
        subset['avg_flops_per_token'],
        subset['acc'],
        color=colors[i],
        linewidth=1.5,
        linestyle='-',
        label=f'prompt_interval={int(p_interval)}',
        zorder=1
    )
    # Plot scatter points for each configuration
    for _, row in subset.iterrows():
        plt.scatter(
            row['avg_flops_per_token'],
            row['acc'],
            marker='o',
            color=colors[i],
            label=row['legend_label'] if row['legend_label'] not in plt.gca().get_legend_handles_labels()[1] else None,
            s=36,
            zorder=2
        )

# Add horizontal dashed line for baseline in 256 steps
if baseline_256_acc is not None:
    plt.axhline(y=baseline_256_acc, color='gray', linestyle='--', alpha=0.5)

plt.title('FLOPs vs Accuracy for Different Settings')
plt.xlabel('Average FLOPs per Token (TFLOPS)')
plt.ylabel('Accuracy')
plt.ylim(bottom=0)
plt.grid(True, alpha=0.3)
plt.legend()
output_path_flops = os.path.join(output_dir, 'flops_vs_acc_line_plot.svg')
plt.savefig(output_path_flops)
plt.close()
print(f"FLOPs vs Accuracy plot saved to {output_path_flops}")

# Plot 2: Run Time vs Accuracy
plt.figure(figsize=(10, 6))

# Plot baseline settings with connected lines and individual legend entries
if all(label in df['legend_label'].values for label in baseline_labels):
    baseline_subset = df[df['legend_label'].isin(baseline_labels)]
    baseline_subset = baseline_subset.sort_values(by='steps')
    plt.plot(
        baseline_subset['arg_run_time_per_token'],
        baseline_subset['acc'],
        color='blue',
        linewidth=2,
        linestyle='-',
        label='baseline',
        zorder=1
    )
    for label in baseline_labels:
        subset = df[df['legend_label'] == label]
        if not subset.empty:
            plt.scatter(
                subset['arg_run_time_per_token'],
                subset['acc'],
                marker='o',
                label=label,
                s=36,
                zorder=2
            )

# Plot non-baseline settings with lines connecting same prompt_interval_steps
for i, p_interval in enumerate(prompt_intervals):
    subset = df[df['prompt_interval_steps'] == p_interval]
    subset = subset.sort_values(by='arg_run_time_per_token')  # Sort to ensure proper line connection
    plt.plot(
        subset['arg_run_time_per_token'],
        subset['acc'],
        color=colors[i],
        linewidth=1.5,
        linestyle='-',
        label=f'prompt_interval={int(p_interval)}',
        zorder=1
    )
    # Plot scatter points for each configuration
    for _, row in subset.iterrows():
        plt.scatter(
            row['arg_run_time_per_token'],
            row['acc'],
            marker='o',
            color=colors[i],
            label=row['legend_label'] if row['legend_label'] not in plt.gca().get_legend_handles_labels()[1] else None,
            s=36,
            zorder=2
        )

# Add horizontal dashed line for baseline in 256 steps
if baseline_256_acc is not None:
    plt.axhline(y=baseline_256_acc, color='gray', linestyle='--', alpha=0.5)

plt.title('Run Time vs Accuracy for Different Settings')
plt.xlabel('Average Run Time per Token (seconds)')
plt.ylabel('Accuracy')
plt.ylim(bottom=0)
plt.grid(True, alpha=0.3)
plt.legend()
output_path_time = os.path.join(output_dir, 'time_vs_acc_line_plot.svg')
plt.savefig(output_path_time)
plt.close()
print(f"Run Time vs Accuracy plot saved to {output_path_time}")