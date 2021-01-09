import configs
from modules.agent import AgentModule
from modules.game import GameModule
import torch
import numpy as np

agent = torch.load('latest.pt')
agent.train(False)

for i in range(10):
    agent.reset()

    config = configs.default_game_config._replace(batch_size=1)

    num_agents = np.random.randint(2, 3+1)
    num_landmarks = np.random.randint(3, 3+1)
    game = GameModule(config, num_agents, num_landmarks)

    _, timesteps = agent(game)

    '''
    This visualizes the trajectories of agents (circles) and target locations (crosses).
    It also displays the communication symbol usage. Basically, alpha channel of a letter represents
    how much the the agent was using the i-th symbol during the epoch (on each step
    communication is done by a [1, 20] float vector). I sum all these vectors through all steps.
    '''
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])
    colors = ['red', 'green', 'blue']
    agent_markers = ['o', '^']
    landmark_markers = ['P', '*']
    utterances = np.zeros_like(timesteps[0]['utterances'][0].detach())
    for time, timestep in enumerate(timesteps):
        agent_legends = []
        for idx, point in enumerate(timestep['locations'][0][:num_agents]):
            agent_legends.append(
            plt.scatter(*list(point.detach().numpy()), 
                    color=colors[int(game.physical[0, idx, 0].item())], 
                    marker=agent_markers[int(game.physical[0, idx, 1].item())],
                    s=20, alpha=0.75)
            )
        for idx, point in enumerate(timestep['locations'][0][-num_landmarks:]):
            if time == 0:
                plt.scatter(*list(point.detach().numpy()), 
                            color='dark'+colors[int(game.physical[0, idx, 0].item())], 
                            marker=landmark_markers[int(game.physical[0, idx, 1].item())],
                            s=300, alpha=0.75)
        utterances += timestep['utterances'][0].detach().numpy()
    # this controls how much we highlight or supress non-freqent symbol when displaying
    # pow < 1 helps to bring in the low freqent symbols that were emitted once and lost in sum
    # pow >=1 can highlight some important symbols through the epoch if it is too noisy
    utterances = np.power(utterances / utterances.max(axis=1)[..., np.newaxis], 0.25)
    for agent_idx in range(utterances.shape[0]):
        for symbol_idx in range(utterances.shape[1]):
            plt.text(0, 1 + 0.01 + 0.05 * agent_idx, str(agent_idx + 1) + ': ',
                    color=colors[int(game.physical[0, agent_idx, 0].item())],
                    transform=ax.transAxes)
            plt.text(0.05 + 0.03 * symbol_idx, 1 + 0.01 + 0.05 * agent_idx, 
                    'ABCDEFGHIJKLMNOPQRSTUVXYZ1234567890'[symbol_idx], 
                    alpha=utterances[agent_idx, symbol_idx], 
                    color=colors[int(game.physical[0, agent_idx, 0].item())],
                    transform=ax.transAxes)
    plt.legend(reversed(agent_legends), reversed([str(i + 1) for i in range(len(agent_legends))]),
                bbox_to_anchor=(0, 1.15))
    plt.show()