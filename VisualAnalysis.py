import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set export folder explicitly
EXPORT_PATH = os.path.join(os.path.dirname(__file__), "export")

dfs = {
    'explorative': pd.read_csv(os.path.join(EXPORT_PATH, "explorative_trials.csv")),
    'recent_k5': pd.read_csv(os.path.join(EXPORT_PATH, "recent_k5_trials.csv")),
    'threshold': pd.read_csv(os.path.join(EXPORT_PATH, "threshold_trials.csv")),
    'maximin': pd.read_csv(os.path.join(EXPORT_PATH, "maximin_trials.csv"))
}

conclusions = pd.read_csv("export/conclusions.csv")
print("Current working dir:", os.getcwd())

colors = {
    'threshold': '#d62728',        # Red
    'maximin': '#2ca02c',          # Green
    'explorative': '#ff7f0e',      # Orange
    'recent_k5': '#1f77b4'         # Blue
}


#####################################
# Plot cumulative reward over time

plt.figure(figsize=(6, 8))
final_rewards = {}

for name, df in dfs.items():
    cum_reward = df.groupby('step')['reward'].mean().cumsum()
    plt.plot(cum_reward, label=name, color=colors[name], linewidth=2.5)
    final_rewards[name] = cum_reward.iloc[-1]  # stores final reward for label

plt.title("Cumulative Benefit Over Time (Averaged Over Runs)")
plt.xlabel("Step")
plt.ylabel("Cumulative Reward")
plt.legend()
plt.grid(True)

# adds a box with final values
text = "\n".join(
    f"{agent}: {reward:.2f}" for agent, reward in sorted(final_rewards.items(), key=lambda x: -x[1])
)
plt.gcf().text(0.75, 0.2, f"Final rewards:\n{text}", fontsize=10,
               bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

plt.tight_layout()
plt.show()

#####################################
#Average Total Reward per Agent

avg_rewards = conclusions.groupby("agent")["total_reward"].mean().reset_index()

plt.figure(figsize=(6, 5))
ax = sns.barplot(
    data=avg_rewards,
    x="agent",
    y="total_reward",
    palette=colors
)

plt.title("Average Total Reward per Agent")
plt.ylabel("Total Reward")
plt.xlabel("Agent")

# Add numbers on bars
for i, row in avg_rewards.iterrows():
    plt.text(i, row["total_reward"] + 15 , f"{row['total_reward']:.2f}", ha='center')

plt.tight_layout()
plt.show()

#####################################
# Boxplot for last 10-step average rewards

# Plot boxplot
agent_order = sorted(conclusions['agent'].unique())
stats = conclusions.groupby("agent")["last_10_avg_reward"].agg(['mean', 'std']).reindex(agent_order)

plt.figure(figsize=(7, 5))
sns.boxplot(
    data=conclusions,
    x="agent",
    y="last_10_avg_reward",
    palette=colors,
    order=agent_order
)

# Overlay expectation (mean) with error bars for std (variance)
for i, agent in enumerate(agent_order):
    mean = stats.loc[agent, 'mean']
    std = stats.loc[agent, 'std']

    # Mean marker
    plt.scatter(i, mean, color='black', zorder=5, label='Mean' if i == 0 else "")

    # Variance marker as vertical bar
    plt.errorbar(i, mean, yerr=std, fmt='none', ecolor='black', capsize=5, linewidth=1, label='± Std' if i == 0 else "")

# Legend only once
plt.legend()
plt.title("Last 10 Steps: Boxplot with Mean ± Variance Overlay")
plt.xlabel("Agent")
plt.ylabel("Avg Reward (Last 10 Steps)")
plt.grid(True)
plt.tight_layout()
plt.show()


#####################################
#Average Risk Frequency (as Percent)

risk_rates = []
for name, df in dfs.items():
    pct_risky = (df['action'] == 'R').mean()
    risk_rates.append({'agent': name, 'risky_rate': pct_risky * 100})  # convert to percent

risk_df = pd.DataFrame(risk_rates)

plt.figure(figsize=(6, 5))
sns.barplot(
    data=risk_df,
    x="agent",
    y="risky_rate",
    palette=[colors[agent] for agent in risk_df["agent"]],
)

plt.title("Average Risk Frequency per Agent")
plt.ylabel("Risky Choices (%)")
plt.xlabel("Agent")


for i, row in risk_df.iterrows():
    plt.text(i, row["risky_rate"] + 0.7, f"{row['risky_rate']:.1f}%", ha='center') # adds % labels on top

plt.ylim(0, 105)
plt.tight_layout()
plt.show()

#####################################
#Change in Average Reward Over Time

# Compute average reward per step for each agent
benefit_change = {}
for agent, df in dfs.items():
    avg_reward_per_step = df.groupby('step')['reward'].mean()
    benefit_change[agent] = avg_reward_per_step.diff().fillna(0)  # change in reward per step

# Combine into a single DataFrame for plotting
benefit_df = pd.DataFrame(benefit_change)

# --- Create subplots for delta benefit per agent ---
fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
axs = axs.flatten()  # Flatten to iterate easily

for i, agent in enumerate(benefit_df.columns):
    axs[i].plot(benefit_df[agent], color=colors[agent], linewidth=1.5)
    axs[i].set_title(f"{agent} - Δ Benefit per Step")
    axs[i].set_xlabel("Step")
    axs[i].set_ylabel("Δ Reward")
    axs[i].grid(True)

plt.suptitle("Change in Average Reward Over Time by Agent", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

#####################################
#Step-by-Step Risk Frequency - Subplot for all 4 agents
fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
axs = axs.flatten()

# For each agent, calculate and plot % of R choices over time
for idx, (agent, df) in enumerate(dfs.items()):
    # % of 'R' at each step across all trials
    step_risk = df.groupby('step')['action'].apply(lambda x: np.mean(x == 'R')) * 100
    axs[idx].plot(step_risk.index, step_risk.values, color=colors[agent], linewidth=2)
    axs[idx].set_title(f"{agent} – Risky Choice % Over Time")
    axs[idx].set_xlabel("Step")
    axs[idx].set_ylabel("Risk %")
    axs[idx].set_ylim(0, 105)
    axs[idx].grid(True)

plt.suptitle("Percentage of 'R' Actions per Step (Each Agent)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()