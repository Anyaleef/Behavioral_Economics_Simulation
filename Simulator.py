import pandas as pd
from typing import List
from cableCart import CableCarEpisode
from Agent import Agent
import os

class BatchSimulator:
    def __init__(self, env_episode: CableCarEpisode, agents: List[Agent], export_to):
        self.env_episode = env_episode
        self.agents = agents
        self.export_dir = export_to

    def run_batch(self, steps: int, runs: int):
        os.makedirs(self.export_dir, exist_ok = True)

        agent_results = {agent.name: [] for agent in self.agents}
        all_conclusions = []  # To store all agents' conclusions

        for run_id in range(runs):
            print("Starting run: ", run_id)
            episode = self.env_episode.generate_episode(steps)

            for agent in self.agents:
                agent.reset()

                for step, outcome in enumerate(episode):
                    action = agent.decide(step)
                    reward = outcome['safe_reward'] if action == 'S' else outcome['risky_reward']
                    agent.observe(action, reward)

                # Save with counterfactuals
                for i, entry in enumerate(agent.history):
                    agent_results[agent.name].append({
                        "run_id": run_id,
                        "step": i,
                        "action": entry["action"],
                        "reward": entry["reward"],
                        "forgone": episode[i]["risky_reward"] if entry["action"] == "S" else episode[i]["safe_reward"]
                    })

        # Export agent-specific log files
        for agent_name, records in agent_results.items():
            df = pd.DataFrame(records)
            df.to_csv(f"{self.export_dir}/{agent_name}_trials.csv", index = False)

            # --- Base aggregation per run ---
            base_summary = df.groupby("run_id").agg(
                total_reward = ('reward', 'sum'),
                avg_reward = ('reward', 'mean'),
                num_safe = ('action', lambda x: (x == 'S').sum()),
                num_risky = ('action', lambda x: (x == 'R').sum()),
            ).reset_index()

            # --- Compute last 10 steps' avg reward per run ---
            last_10 = (
                df.groupby("run_id")["reward"]
                .apply(lambda x: x.iloc[-10:].mean())
                .reset_index(name = "last_10_avg_reward")
            )
            # --- Merge ---
            summary = pd.merge(base_summary, last_10, on = "run_id")
            summary['agent'] = agent_name
            all_conclusions.append(summary)

        # Combine all agents' conclusions into one file
        combined_conclusions = pd.concat(all_conclusions, ignore_index = True)
        combined_conclusions.to_csv(f"{self.export_dir}/conclusions.csv", index = False)
        print(f"Exporting results to {self.export_dir}")
        return agent_results