import pandas as pd
import os
import matplotlib.pyplot as plt
# Folder containing results
results_dir = "learning_rates"

# Loop through all learning rate folders
for lr_folder in sorted(os.listdir(results_dir)):
    folder_path = os.path.join(results_dir, lr_folder)

    # Extract learning rate from folder name
    try:
        learning_rate = float(lr_folder.split('_')[1])
    except (IndexError, ValueError):
        print(f"Skipping folder '{lr_folder}' - Invalid format.")
        continue

    print(f"Results for Learning Rate {learning_rate}:")

    # Process episode_stats.csv
    stats_file = os.path.join(folder_path, "episode_stats.csv")
    if os.path.exists(stats_file):
        stats = pd.read_csv(stats_file)
        avg_reward = stats["reward_per_episode"].mean()
        avg_failures = stats["match_fail_count"].mean()
        total_failures = stats["match_fail_count"].sum()
        convergence_episode = stats[stats["epsilon"] == stats["epsilon"].min()]["Episode"].min()
        print(f"  Average Reward: {avg_reward:.2f}")
        print(f"  Average Failures: {avg_failures:.2f}")
        print(f"  Total Failures: {total_failures}")
        print(f"  Convergence Episode: {convergence_episode}")
    else:
        print("  episode_stats.csv not found.")

    # Process q_tables.csv
    q_table_file = os.path.join(folder_path, "q_tables.csv")
    if os.path.exists(q_table_file):
        q_table = pd.read_csv(q_table_file)
        max_q_value = q_table["Q-Value"].max()
        avg_q_value = q_table["Q-Value"].mean()
        print(f"  Max Q-Value: {max_q_value:.2f}")
        print(f"  Average Q-Value: {avg_q_value:.2f}")
    else:
        print("  q_tables.csv not found.")

    print()  # Blank line for better readability

# Data from your results
learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
avg_rewards = [4.69, 4.55, 4.45, 4.46, 4.46, 4.46, 4.46, 4.46, 4.46]
avg_failures = [15.54, 15.67, 15.78, 15.77, 15.77, 15.77, 15.77, 15.77, 15.77]
convergence_episodes = [4556] * len(learning_rates)

# Plot Average Reward
plt.figure(figsize=(10, 6))
plt.plot(learning_rates, avg_rewards, marker='o', label='Average Reward')
plt.title('Average Reward vs Learning Rate')
plt.xlabel('Learning Rate')
plt.ylabel('Average Reward')
plt.grid()
plt.legend()
plt.show()

# Plot Average Match Failures
plt.figure(figsize=(10, 6))
plt.plot(learning_rates, avg_failures, marker='o', color='red', label='Average Match Failures')
plt.title('Average Match Failures vs Learning Rate')
plt.xlabel('Learning Rate')
plt.ylabel('Average Match Failures')
plt.grid()
plt.legend()
plt.show()
    
