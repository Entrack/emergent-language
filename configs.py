from typing import NamedTuple, Any, List
import numpy as np
import constants

DEFAULT_NUM_EPOCHS = 50000
DEFAULT_LR = 1e-4

DEFAULT_HIDDEN_SIZE = 128
DEFAULT_DROPOUT = 0.1
DEFAULT_FEAT_VEC_SIZE = 128
DEFAULT_TIME_HORIZON = 32

USE_UTTERANCES = True
PENALIZE_WORDS = False
DEFAULT_VOCAB_SIZE = 6
DEFAULT_OOV_PROB = 5

DEFAULT_WORLD_DIM = 16
MAX_AGENTS = 3
MAX_LANDMARKS = 3
USE_STRICT_COLORS = True
STRICT_COLORS = np.array([[constants.COLOR_SCALE, 0, 0], [0, constants.COLOR_SCALE, 0], [0, 0, constants.COLOR_SCALE]])
USE_SHAPES = True
NUM_SHAPES = 2

TrainingConfig = NamedTuple('TrainingConfig', [
    ('num_epochs', int),
    ('learning_rate', float)
    ])

GameConfig = NamedTuple('GameConfig', [
    ('world_dim', Any),
    ('max_agents', int),
    ('max_landmarks', int),
    ('use_strict_colors', bool),
    ('strict_colors', Any),
    ('use_shapes', bool),
    ('num_shapes', int),
    ('use_utterances', bool),
    ('vocab_size', int),
    ('memory_size', int)
])

ProcessingModuleConfig = NamedTuple('ProcessingModuleConfig', [
    ('input_size', int),
    ('hidden_size', int),
    ('dropout', float)
    ])

WordCountingModuleConfig = NamedTuple('WordCountingModuleConfig', [
    ('vocab_size', int),
    ('oov_prob', float)
    ])

GoalPredictingProcessingModuleConfig = NamedTuple("GoalPredictingProcessingModuleConfig", [
    ('processor', ProcessingModuleConfig),
    ('hidden_size', int),
    ('dropout', float),
    ('goal_size', int)
    ])

ActionModuleConfig = NamedTuple("ActionModuleConfig", [
    ('goal_processor', ProcessingModuleConfig),
    ('action_processor', ProcessingModuleConfig),
    ('hidden_size', int),
    ('dropout', float),
    ('movement_dim_size', int),
    ('movement_step_size', int),
    ('vocab_size', int),
    ('use_utterances', bool)
    ])

AgentModuleConfig = NamedTuple("AgentModuleConfig", [
    ('time_horizon', int),
    ('feat_vec_size', int),
    ('movement_dim_size', int),
    ('goal_size', int),
    ('vocab_size', int),
    ('utterance_processor', GoalPredictingProcessingModuleConfig),
    ('physical_processor', ProcessingModuleConfig),
    ('action_processor', ActionModuleConfig),
    ('word_counter', WordCountingModuleConfig),
    ('use_utterances', bool),
    ('penalize_words', bool)
    ])

default_training_config = TrainingConfig(
        num_epochs=DEFAULT_NUM_EPOCHS,
        learning_rate=DEFAULT_LR)

default_word_counter_config = WordCountingModuleConfig(
        vocab_size=DEFAULT_VOCAB_SIZE,
        oov_prob=DEFAULT_OOV_PROB)

default_game_config = GameConfig(
        DEFAULT_WORLD_DIM,
        MAX_AGENTS,
        MAX_LANDMARKS,
        USE_STRICT_COLORS,
        STRICT_COLORS,
        USE_SHAPES,
        NUM_SHAPES,
        USE_UTTERANCES,
        DEFAULT_VOCAB_SIZE,
        DEFAULT_HIDDEN_SIZE)

if USE_UTTERANCES:
    feat_size = DEFAULT_FEAT_VEC_SIZE*3
else:
    feat_size = DEFAULT_FEAT_VEC_SIZE*2

def get_processor_config_with_input_size(input_size):
    return ProcessingModuleConfig(
        input_size=input_size,
        hidden_size=DEFAULT_HIDDEN_SIZE,
        dropout=DEFAULT_DROPOUT)

default_action_module_config = ActionModuleConfig(
        goal_processor=get_processor_config_with_input_size(constants.GOAL_SIZE),
        action_processor=get_processor_config_with_input_size(feat_size),
        hidden_size=DEFAULT_HIDDEN_SIZE,
        dropout=DEFAULT_DROPOUT,
        movement_dim_size=constants.MOVEMENT_DIM_SIZE,
        movement_step_size=constants.MOVEMENT_STEP_SIZE,
        vocab_size=DEFAULT_VOCAB_SIZE,
        use_utterances=USE_UTTERANCES)

default_goal_predicting_module_config = GoalPredictingProcessingModuleConfig(
    processor=get_processor_config_with_input_size(DEFAULT_VOCAB_SIZE),
    hidden_size=DEFAULT_HIDDEN_SIZE,
    dropout=DEFAULT_DROPOUT,
    goal_size=constants.GOAL_SIZE)

default_agent_config = AgentModuleConfig(
        time_horizon=DEFAULT_TIME_HORIZON,
        feat_vec_size=DEFAULT_FEAT_VEC_SIZE,
        movement_dim_size=constants.MOVEMENT_DIM_SIZE,
        utterance_processor=default_goal_predicting_module_config,
        physical_processor=get_processor_config_with_input_size(constants.MOVEMENT_DIM_SIZE + constants.PHYSICAL_EMBED_SIZE),
        action_processor=default_action_module_config,
        word_counter=default_word_counter_config,
        goal_size=constants.GOAL_SIZE,
        vocab_size=DEFAULT_VOCAB_SIZE,
        use_utterances=USE_UTTERANCES,
        penalize_words=PENALIZE_WORDS)

def get_training_config(kwargs):
    return TrainingConfig(
            num_epochs=kwargs['n-epochs'] if 'n-epochs' in kwargs else default_training_config.num_epochs,
            learning_rate=kwargs['learning-rate'] if 'learning-rate' in kwargs else default_training_config.learning_rate)

def get_game_config(kwargs):
    return GameConfig(
            world_dim=kwargs['world-dim'] if 'world-dim' in kwargs else default_game_config.world_dim,
            max_agents=kwargs['n-agents'] if 'n-agents' in kwargs else default_game_config.max_agents,
            max_landmarks=kwargs['n-landmarks'] if 'n-landmarks' in kwargs else default_game_config.max_landmarks,
            use_strict_colors=kwargs['use-strict-colors'] if 'use-strict-colors' in kwargs else default_game_config.use_strict_colors,
            strict_colors=default_game_config.strict_colors,
            use_shapes=default_game_config.use_shapes,
            num_shapes=default_game_config.num_shapes,
            use_utterances= (not kwargs['no-utterances']) if 'no-utterances' in kwargs else default_game_config.use_utterances,
            vocab_size=kwargs['vocab-size'] if 'vocab-size' in kwargs else default_game_config.vocab_size,
            memory_size=default_game_config.memory_size
            )

def get_agent_config(kwargs):
    vocab_size = kwargs['vocab-size'] if 'vocab-size' in kwargs else DEFAULT_VOCAB_SIZE
    use_utterances = (not kwargs['no-utterances']) if 'no-utterances' in kwargs else USE_UTTERANCES
    penalize_words = kwargs['penalize-words'] if 'penalize-words' in kwargs else PENALIZE_WORDS
    oov_prob = kwargs['oov-prob'] if 'oov-prob' in kwargs else DEFAULT_OOV_PROB
    if use_utterances:
        feat_vec_size = DEFAULT_FEAT_VEC_SIZE*3
    else:
        feat_vec_size = DEFAULT_FEAT_VEC_SIZE*2
    utterance_processor = GoalPredictingProcessingModuleConfig(
            processor=get_processor_config_with_input_size(vocab_size),
            hidden_size=DEFAULT_HIDDEN_SIZE,
            dropout=DEFAULT_DROPOUT,
            goal_size=constants.GOAL_SIZE)
    action_processor = ActionModuleConfig(
            goal_processor=get_processor_config_with_input_size(constants.GOAL_SIZE),
            action_processor=get_processor_config_with_input_size(feat_vec_size),
            hidden_size=DEFAULT_HIDDEN_SIZE,
            dropout=DEFAULT_DROPOUT,
            movement_dim_size=constants.MOVEMENT_DIM_SIZE,
            movement_step_size=constants.MOVEMENT_STEP_SIZE,
            vocab_size=vocab_size,
            use_utterances=use_utterances)
    word_counter = WordCountingModuleConfig(
            vocab_size=vocab_size,
            oov_prob=oov_prob)

    return AgentModuleConfig(
            time_horizon=kwargs['n-timesteps'] if 'n-timesteps' in kwargs else default_agent_config.time_horizon,
            feat_vec_size=default_agent_config.feat_vec_size,
            movement_dim_size=default_agent_config.movement_dim_size,
            utterance_processor=utterance_processor,
            physical_processor=default_agent_config.physical_processor,
            action_processor=action_processor,
            word_counter=word_counter,
            goal_size=default_agent_config.goal_size,
            vocab_size=vocab_size,
            use_utterances=use_utterances,
            penalize_words=penalize_words
            )

