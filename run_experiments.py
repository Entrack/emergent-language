import argparse
import numpy as np
import torch
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau
import configs
from modules.agent import AgentModule
from modules.game import GameModule
from collections import defaultdict

from matplotlib import pyplot as plt
plt.set_cmap('bone')
plt.rcParams['savefig.dpi'] = 300
import numpy as np
from collections import defaultdict
from os.path import join
from scipy.stats import entropy
from tqdm import tqdm
from pathlib import Path


picture_extension = '.png'

def plot_symbol_sequences(destination_folder, naming, logs, run):
    epochs_to_use = []
    num_epochs = logs['utterances'].shape[1]
    for milestone in [0.25, 0.5, 1]:
        epoch = max(0, int(num_epochs * milestone) - 1)
        if epoch not in epochs_to_use:
            epochs_to_use.append(epoch)

    for epoch in epochs_to_use:
        plt.clf()
        utterances = logs['utterances'][run, epoch]
        num_timesteps = utterances.shape[1]
        vocab_size = utterances.shape[-1]
        symbols = np.argmax(utterances, axis=-1)
        for timestep in range(num_timesteps):
            symbols_freq = np.bincount(symbols[:, timestep, :].flatten(), minlength=vocab_size)
            symbols_freq = symbols_freq / symbols_freq.max()
            symbols_freq = np.power(symbols_freq, 2)
            for symbol in np.arange(vocab_size):
                plt.scatter(timestep, symbol, color='lightblue', 
                            alpha=symbols_freq[symbol])
        plt.xticks(np.arange(num_timesteps), np.arange(num_timesteps) + 1)
        if vocab_size >= 30:
            plt.yticks(np.arange(vocab_size)[1::2], (np.arange(vocab_size) + 1)[1::2])
        else:        
            plt.yticks(np.arange(vocab_size), np.arange(vocab_size) + 1)
        plt.xlabel('Timestep')
        plt.ylabel('Symbol')
        # plt.show()
        plt.savefig(join(destination_folder, f'disp_argmax-run_{run}-' + naming + '-epoch_'
                     + f"{epoch:02d}") + picture_extension, transparent=True)

    for epoch in epochs_to_use:
        plt.clf()
        utterances = logs['utterances'][run, epoch]
        vocab_size = utterances.shape[-1]
        plt.bar(range(vocab_size), [utterances[:, :, :, i].sum() for i in range(vocab_size)], color='cornflowerblue')
        if vocab_size >= 30:
            plt.xticks(np.arange(vocab_size)[1::2], (np.arange(vocab_size) + 1)[1::2])
        else:        
            plt.xticks(np.arange(vocab_size), np.arange(vocab_size) + 1)
        plt.yticks([])
        plt.xlabel('Symbol')
        plt.ylabel('Cumulative value')
        # plt.show()
        plt.savefig(join(destination_folder, f'bar_utterances-run_{run}-' + naming + '-epoch_'
                     + f"{epoch:02d}") + picture_extension, transparent=True)

def plot_trajectories(destination_folder, name, logs, run, epoch, batch):
    locations_agents = logs['locations_agents'][run, epoch, batch]
    locations_landmarks = logs['locations_landmarks'][run, epoch, batch]
    physical_agents = logs['physical_agents'][run, epoch, batch]
    physical_landmarks = logs['physical_landmarks'][run, epoch, batch]
    if 'utterances' in logs:
        utterances = logs['utterances'][run, epoch, batch]

    plt.clf()
    fig, ax = plt.subplots()

    colors = ['firebrick', 'forestgreen', 'dodgerblue']
    agent_markers = ['o', '^']
    landmark_markers = ['P', '*']

    agent_legends = []
    for idx in range(locations_agents.shape[1]):
        agent_legends.append(
        plt.scatter(locations_agents[:, idx, 0], locations_agents[:, idx, 1],
                color=colors[int(physical_agents[0, idx, 0])], 
                marker=agent_markers[int(physical_agents[0, idx, 1])], 
                s=20, alpha=0.75)
        )
    for idx in range(locations_landmarks.shape[1]):
        plt.scatter(locations_landmarks[0, idx, 0], locations_landmarks[0, idx, 1],
                color=colors[int(physical_landmarks[0, idx, 0])], 
                marker=landmark_markers[int(physical_landmarks[0, idx, 1])], 
                s=300, alpha=0.6)

    if 'utterances' in logs:
        utterances_sum = utterances.sum(axis=0)
        utterances_sum = np.power(utterances_sum / utterances_sum.max(axis=1, keepdims=True), 2)
        for agent_idx in range(utterances_sum.shape[0]):
            for symbol_idx in range(utterances_sum.shape[1]):
                plt.text(0, 1 + 0.01 + 0.05 * agent_idx, str(agent_idx + 1) + ': ',
                        color=colors[int(physical_agents[0, agent_idx, 0])],
                        transform=ax.transAxes)
                plt.text(0.05 + 0.03 * symbol_idx, 1 + 0.01 + 0.05 * agent_idx, 
                        'ABCDEFGHIJKLMNOPQRSTUVXYZ1234567890abcde'[symbol_idx], # for more symbols expand this string
                        alpha=utterances_sum[agent_idx, symbol_idx], 
                        color=colors[int(physical_agents[0, agent_idx, 0])],
                        transform=ax.transAxes)

    plt.legend(reversed(agent_legends), reversed([str(i + 1) for i in range(len(agent_legends))]),
            bbox_to_anchor=(0, 1.1265))

    plt.xticks([])
    plt.yticks([])
    
    # plt.show()
    plt.savefig(join(destination_folder, name) + picture_extension, transparent=True)

def get_features_graph(destination_folder, name, values, xlabel='', ylabel=''):
    plt.clf()
    plt.imshow(values, vmin=0.0, vmax=1.0)

    plt.xticks([])
    plt.yticks([])

    # plt.show()
    plt.savefig(join(destination_folder, name) + picture_extension, transparent=True)

def plot_graph_with_std(destination_folder, name, values, xlabel='', ylabel=''):
    plt.clf()
    std_alpha = 0.25

    ticks = np.arange(values.shape[1])
    mean = values.mean(axis=0)
    std = values.std(axis=0)

    plt.plot(mean, color='cornflowerblue')
    plt.fill_between(ticks, mean-std, mean+std, color='cornflowerblue', alpha=std_alpha)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # plt.show()
    plt.savefig(join(destination_folder, name) + picture_extension, transparent=True)

def get_naming(name, value, values):
    naming = str(name) + '_'
    if type(values[0]) not in [int, float]:
        naming += str(value)
    else:
        min_precision = max(0, int(np.log10(1/min(values))))
        naming += f"{value:.{min_precision}f}"
    return naming
    

def visualize_logs(destination_folder, logs, runs_number, parameter_name, parameter_value, values):
    naming = get_naming(parameter_name, parameter_value, values)

    if 'utterances' in logs.keys():
        for run in range(runs_number):
            plot_symbol_sequences(destination_folder, naming, logs, run)

    plot_graph_with_std(destination_folder, 
                        'loss_game-' + naming, logs['loss_game'].sum(axis=2), 
                        'Epoch', 'Game loss')
    plot_graph_with_std(destination_folder, 
                        'loss_physical-' + naming, logs['loss_physical'].sum(axis=2), 
                        'Epoch', 'Physical loss')
    plot_graph_with_std(destination_folder, 
                        'loss_goal_prediction-' + naming, logs['loss_goal_prediction'].sum(axis=2), 
                        'Epoch', 'Goal prediction loss')

    if 'loss_word_count' in logs.keys():
        plot_graph_with_std(destination_folder, 
                        'loss_word_count-' + naming, logs['loss_word_count'].sum(axis=2), 
                        'Epoch', 'Rare words loss')
    if 'speech_entropy' in logs.keys():
        plot_graph_with_std(destination_folder, 
                        'speech_entropy-' + naming, logs['speech_entropy'].sum(axis=2), 
                        'Epoch', 'Speech entropy')

    if 'utterances' in logs.keys():
        plot_graph_with_std(destination_folder, 
                        'speech_entropy_normalized-' + naming, entropy(logs['utterances'], axis=-1).mean(axis=tuple(range(2, 5))), 
                        'Epoch', 'Speech entropy normalized')


    plot_trajectories(destination_folder, 'trajectories-epoch_01_batch_0-' + naming, logs, 0, 0, 0)
    plot_trajectories(destination_folder, 'trajectories-epoch_01_batch_1-' + naming, logs, 0, 0, 1)

    for batch_index in range(5):
        plot_trajectories(destination_folder, 
                        f'trajectories-epoch_fi_batch_{batch_index}-' + naming, logs, 0, -1, batch_index)

def train_one_epoch(epoch,
        agent_config, game_config, training_config,
        agent, optimizer, scheduler,
        losses, dists):

    num_agents = np.random.randint(game_config.min_agents, game_config.max_agents+1)
    num_landmarks = np.random.randint(game_config.min_landmarks, game_config.max_landmarks+1)
    agent.reset()
    game = GameModule(game_config, num_agents, num_landmarks)
    if training_config.use_cuda:
        game.cuda()
    optimizer.zero_grad()

    total_loss, epoch_logs = agent(game, new_log_format=True)
    per_agent_loss = total_loss.data[0] / num_agents / game_config.batch_size
    losses[num_agents][num_landmarks].append(per_agent_loss)

    total_loss.backward()
    optimizer.step()

    if num_agents == game_config.max_agents and num_landmarks == game_config.max_landmarks:
        scheduler.step(losses[game_config.max_agents][game_config.max_landmarks][-1])

    return epoch_logs

def train(kwargs):
    args = defaultdict(lambda: False, kwargs)
    agent_config = configs.get_agent_config(args)
    game_config = configs.get_game_config(args)
    training_config = configs.get_training_config(args)
    agent = AgentModule(agent_config)
    if training_config.use_cuda:
        agent.cuda()
    optimizer = RMSprop(agent.parameters(), lr=training_config.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, cooldown=5)
    losses = defaultdict(lambda:defaultdict(list))
    dists = defaultdict(lambda:defaultdict(list))

    run_logs = defaultdict(lambda: list())

    # for epoch in tqdm(range(training_config.num_epochs), desc='Epochs'):
    for epoch in range(training_config.num_epochs):
        epoch_logs = train_one_epoch(epoch,
                        agent_config, game_config, training_config,
                        agent, optimizer, scheduler,
                        losses, dists)
        for name, values in epoch_logs.items():
            run_logs[name].append(values)
    return run_logs

def run_training_n_times(runs_number, kwargs):
    logs = defaultdict(lambda: list())
    # for run_number in tqdm(range(runs_number), desc='Runs'):
    for run_number in range(runs_number):
        run_logs = train(kwargs)
        for name, values in run_logs.items():
            logs[name].append(values)
    logs = {name : np.array(values) for name, values in logs.items()}
    logs = {name : np.moveaxis(values, 3, 2) if values.ndim == 6 else values for name, values in logs.items()}
    return logs

def args_to_str(args):
    string = ""
    args_list = []
    for key in sorted(args):
        args_list.append(key)
        args_list.append(str(args[key]))
    string = "_".join(args_list)
    return string

def create_experiment_folder(destination_folder, runs_number, parameter_name, parameter_values, kwargs):
    experiment_path = join(destination_folder, 
                            '_'.join([str(el) for el in 
                            ['runs', runs_number, parameter_name, parameter_values, args_to_str(kwargs)]]))
    Path(experiment_path).mkdir(parents=True, exist_ok=True)
    return experiment_path

def main():
    for configuration in tqdm(experiments['configurations'], desc='Confugurations'): 

        for name, values in tqdm(configuration['parameters_values'].items(), desc='Parameters'):
            experiment_folder = create_experiment_folder(destination_folder, runs_number, name, values, configuration['kwargs'])
            for value in tqdm(values, desc='Values'):

                logs = run_training_n_times(runs_number, 
                                            {** {**experiments['default_kwargs'], **configuration['kwargs']}, 
                                             ** {name: value}})
                visualize_logs(experiment_folder, logs, runs_number, name, value, values)



destination_folder = './experiments'
experiments = {
    'configurations': [
        {
            'parameters_values': {
                'control_experiment': ['!']
            },
            'kwargs': {}
        },
        # {
        #     'parameters_values': {
        #         'oov_prob': [0.25, 0.5, 1.5]
        #     },
        #     'kwargs': {
        #         'penalize_words': True,
        #     }
        # },
        # {
        #     'parameters_values': {
        #         'vocab_size': [2, 4, 10, 36],
        #     },
        #     'kwargs': {}
        # },
        # {
        #     'parameters_values': {
        #         'no_utterances': [True],
        #     },
        #     'kwargs': {},
        # },
        # {
        #     'para   meters_values': {
        #         'use_lstm': [True],
        #     },
        #     'kwargs': {}
        # },
        # {
        #     'parameters_values': {
        #         'entropy_normalization': [None, "batch", "agent", "symbol"],
        #     },
        #     'kwargs': {
        #         'penalize_entropy': True,
        #         'entropy_weight': 20,
        #     }
        # },
    ],
    'default_kwargs': {
        'max_agents': 2, # to be able to convert logs to array
        'n_epochs': 50,
    }
}
runs_number = 2 # to produce graphs with std intervals

if __name__ == "__main__":
    main()
