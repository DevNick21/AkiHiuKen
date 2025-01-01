from q_learning import main
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Ask user if all data has been collected
# Because the process of collecting data can take a long time, and shouldn't be repeated if not necessary
while True:
    check = str(input("Has all the data been collected? Enter yes/no: ")).lower()
    if check in ['yes', 'no']:
        break
    elif check == 'stop':
        exit()
    else:
        print("Invalid input. Please enter 'yes' or 'no'.")


if check == 'no':
    averages = main()
    columns = ['average_reward_per_episode',
               'average_match_fail_count', 'average_time_per_episode']

    for column in columns:
        column_name = column.split('_')
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=averages, x='learning_rate',
                     y=f'{column}', hue='discount_factor')
        plt.title(
            f'Average {" ".join(column_name[1:]).capitalize()} by Learning Rate and Discount Factor')
        plt.xlabel('Learning Rate')
        plt.ylabel(f'Average {" ".join(column_name[1:]).capitalize()}')
        plt.savefig(f'plots/Average {"_".join(column_name[1:]).capitalize()}')
        plt.show()
elif check == 'yes':
    best_df = pd.read_csv(
        'learning_rate_0.1/discount_factor_0.1/episode_stats.csv')
    plt.figure(figsize=(10, 6))
    plt.scatter(best_df['epsilon'], best_df['match_fail_count'],
                alpha=0.5, s=10, color='r')
    plt.title('Epsilon vs Total Match Fail Count')
    plt.xlabel('Epsilon')
    plt.ylabel('Reward')
    plt.savefig('plots/Total_match_fail_Count_for_best')
    plt.grid(True)
    plt.show()
