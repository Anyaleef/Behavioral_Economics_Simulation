from cableCart import CableCarEpisode
from Agent import ExplorativeAgent, SmallSampleAgent, ThresholdAgent, MaximinAgent
from Simulator import BatchSimulator

def utility_function(T: int) -> float:
    """
    Example utility function for the risky path.
    This could be any function of T (the number of skipped cars).
    """
    return -0.15*(T)**2


if __name__ == "__main__":
    env = CableCarEpisode(p_empty = 0.1, reward_for_empty = 25 , discomfort_mean = 5, discomfort_std = 2, f = utility_function)
    agents = [
        ExplorativeAgent(name ="explorative"),
        SmallSampleAgent(K = 5, name = "recent_k5"),
        ThresholdAgent(name = "threshold"),
        MaximinAgent(name="maximin")
    ]
    import os
    print("Current working directory:", os.getcwd())
    #export_to = os.getcwd()+r"\export"
    export_to = os.path.join(os.path.dirname(__file__), "export")
    sim = BatchSimulator(env, agents, export_to)
    sim.run_batch(steps=300, runs=200)
