import torch
import torch.nn as nn
from torch.autograd import Variable

from modules.processing import ProcessingModule
from modules.goal_predicting import GoalPredictingProcessingModule
from modules.action import ActionModule
from modules.word_counting import WordCountingModule

from collections import defaultdict


"""
    The AgentModule is the general module that's responsible for the execution of
    the overall policy throughout training. It holds all information pertaining to
    the whole training episode, and at each forward pass runs a given game until
    the end, returning the total cost all agents collected over the entire game
"""
class AgentModule(nn.Module):
    def __init__(self, config):
        super(AgentModule, self).__init__()
        self.init_from_config(config)
        self.total_cost = Variable(self.Tensor(1).zero_())

        self.physical_processor = ProcessingModule(config.physical_processor)
        self.physical_pooling = nn.AdaptiveMaxPool2d((1, config.feature_vec_size))
        self.action_processor = ActionModule(config.action_processor)

        if self.using_utterances:
            self.utterance_processor = GoalPredictingProcessingModule(config.utterance_processor)
            self.utterance_pooling = nn.AdaptiveMaxPool2d((1, config.feature_vec_size))
            if self.penalizing_words:
                self.word_counter = WordCountingModule(config.word_counter)

    def init_from_config(self, config):
        self.training = True
        self.using_utterances = config.use_utterances
        self.penalizing_words = config.penalize_words
        self.using_cuda = config.use_cuda
        self.time_horizon = config.time_horizon
        self.movement_dim_size = config.movement_dim_size
        self.vocab_size = config.vocab_size
        self.goal_size = config.goal_size
        self.processing_hidden_size = config.physical_processor.hidden_size
        self.Tensor = torch.cuda.FloatTensor if self.using_cuda else torch.FloatTensor

        self.penalizing_speech_entropy = config.penalize_entropy
        self.entropy_weight = config.entropy_weight
        self.entropy_normalization = config.entropy_normalization

    def reset(self):
        self.total_cost = torch.zeros_like(self.total_cost)
        if self.using_utterances and self.penalizing_words:
            self.word_counter.word_counts = torch.zeros_like(self.word_counter.word_counts)

    def train(self, mode=True):
        super(AgentModule, self).train(mode)
        self.training = mode

    def update_mem(self, game, mem_str, new_mem, agent, other_agent=None):
        # TODO: Look into tensor copying from Variable
        new_big_mem = Variable(self.Tensor(game.memories[mem_str].data))
        if other_agent is not None:
            new_big_mem[:, agent, other_agent] = new_mem
        else:
            new_big_mem[:, agent] = new_mem
        game.memories[mem_str] = new_big_mem

    def process_utterances(self, game, agent, other_agent, utterance_processes, goal_predictions):
        utterance_processed, new_mem, goal_predicted = self.utterance_processor(game.utterances[:,other_agent], game.memories["utterance"][:, agent, other_agent])
        self.update_mem(game, "utterance", new_mem, agent, other_agent)
        utterance_processes[:, other_agent, :] = utterance_processed
        goal_predictions[:, agent, other_agent, :] = goal_predicted

    def process_physical(self, game, agent, other_entity, physical_processes):
        physical_processed, new_mem = self.physical_processor(torch.cat((game.observations[:,agent,other_entity],game.physical[:,other_entity]), 1), game.memories["physical"][:,agent, other_entity])
        self.update_mem(game, "physical", new_mem,agent, other_entity)
        physical_processes[:,other_entity,:] = physical_processed

    def get_physical_feature(self, game, agent):
        physical_processes = Variable(self.Tensor(game.batch_size, game.num_entities, self.processing_hidden_size))
        for entity in range(game.num_entities):
            self.process_physical(game, agent, entity, physical_processes)
        return self.physical_pooling(physical_processes)

    def get_utterance_feature(self, game, agent, goal_predictions):
        if self.using_utterances:
            utterance_processes = Variable(self.Tensor(game.batch_size, game.num_agents, self.processing_hidden_size))
            for other_agent in range(game.num_agents):
                self.process_utterances(game, agent, other_agent, utterance_processes, goal_predictions)
            return self.utterance_pooling(utterance_processes)
        else:
            return None

    def get_action(self, game, agent, physical_feature, utterance_feature, movements, utterances):
        movement, utterance, new_mem = self.action_processor(physical_feature, game.observed_goals[:,agent], game.memories["action"][:,agent], self.training, utterance_feature)
        self.update_mem(game, "action", new_mem, agent)
        movements[:,agent,:] = movement
        if self.using_utterances:
            utterances[:,agent,:] = utterance

    def forward(self, game, new_log_format=False):
        if not new_log_format:
            epoch_logs = []
        else:
            epoch_logs = defaultdict(lambda: list())
            
        for t in range(self.time_horizon):
            # Batch size represents how many games would be run in parallel
            movements = Variable(self.Tensor(game.batch_size, game.num_entities, self.movement_dim_size).zero_())
            utterances = None
            goal_predictions = None
            if self.using_utterances:
                utterances = Variable(self.Tensor(game.batch_size, game.num_agents, self.vocab_size))
                # What does each agent think of anothers landmark probality distribution
                goal_predictions = Variable(self.Tensor(game.batch_size, game.num_agents, game.num_agents, self.goal_size))

            for agent in range(game.num_agents):
                physical_feature = self.get_physical_feature(game, agent)
                utterance_feature = self.get_utterance_feature(game, agent, goal_predictions)
                self.get_action(game, agent, physical_feature, utterance_feature, movements, utterances)

            costs = game(movements, goal_predictions, utterances, new_log_format)

            if not new_log_format:
                costs_sum = costs
            else:
                costs_sum = sum(costs)
                loss_game = costs_sum.clone().detach()

            if self.penalizing_words:
                word_count_cost = self.word_counter(utterances)
                costs_sum = costs_sum + word_count_cost
            
            if self.penalizing_speech_entropy:

                # def normalize_by_axis(tensor, axis, exclude=[]):
                #     axis_indices = [i for i in list(range(tensor.dim())) if not i == axis and not i in exclude]
                #     return tensor / tensor.sum(axis=axis_indices, keepdims=True), axis_indices
                
                if self.entropy_normalization == None:
                    normalized = utterances / utterances.sum()
                    speech_entropy = (-normalized * torch.log(normalized)).sum() * self.entropy_weight
                else:
                    if self.entropy_normalization == 'batch':
                        entropy_minimization_axis = 0 # (batches, agents, symbols)
                    if self.entropy_normalization == 'agent':
                        entropy_minimization_axis = 1 # (batches, agents, symbols)
                    if self.entropy_normalization == 'symbol':
                        entropy_minimization_axis = 2 # (batches, agents, symbols)

                    get_axis_except = lambda x, i: [j for j in list(range(x.dim())) if not j == i]
                    normalized = utterances / utterances.sum(axis=entropy_minimization_axis, keepdims=True)
                    excluded_axis = get_axis_except(utterances, entropy_minimization_axis)
                    normalized_summed = normalized.sum(axis=excluded_axis)
                    speech_entropy = -(normalized_summed * torch.log(normalized_summed)).sum() * self.entropy_weight

                    # normalized, excluded_dims = normalize_by_axis(utterances, entropy_minimization_axis)
                    # # speech_entropy = (normalized.sum(axis=2) * torch.log(normalized.sum(axis=2))).sum() * self.entropy_weight
                    # speech_entropy = (normalized.sum(axis=entropy_minimization_axis) 
                    #                 * torch.log(normalized.sum(axis=entropy_minimization_axis))).sum() * self.entropy_weight
        
                costs_sum = costs_sum + speech_entropy

            self.total_cost = self.total_cost + costs_sum
            if not new_log_format:
                epoch_logs.append({
                    'locations': game.locations,
                    'loss_total': costs,
                    })
                if self.using_utterances:
                    epoch_logs[-1]['utterances'] = utterances
            else:
                epoch_logs['loss_game'].append(loss_game.detach().item())
                epoch_logs['loss_physical'].append(costs[0].detach().item())
                epoch_logs['loss_goal_prediction'].append(costs[1].detach().item())

                if self.penalizing_words:
                    epoch_logs['loss_word_count'].append(word_count_cost.detach().numpy())
                if self.penalizing_speech_entropy:
                    epoch_logs['speech_entropy'].append(speech_entropy.detach().numpy())
                
                epoch_logs['locations_agents'].append(game.locations[:, :game.num_agents, :].detach().numpy())
                epoch_logs['physical_agents'].append(game.physical[:, :game.num_agents, :].detach().numpy())
                epoch_logs['locations_landmarks'].append(game.locations[:, game.num_agents:, :].detach().numpy())
                epoch_logs['physical_landmarks'].append(game.physical[:, game.num_agents:, :].detach().numpy())
                if self.using_utterances:
                    epoch_logs['utterances'].append(utterances.detach().numpy())
        return self.total_cost, epoch_logs
