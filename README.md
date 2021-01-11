# emergent-language
An implementation of Emergence of Grounded Compositional Language in Multi-Agent Populations by Igor Mordatch and Pieter Abbeel

## Installation
(in this folder)
```
virutalenv -p python3 .venv
source .venv/Scripts/activate # for Windows
soruce .venv/bin/activate # for Linux
pip install -r requirements.txt
```
Then everything should be ready to go.

## Default README follows

To run, invoke `python3 train.py` in environment with PyTorch installed. To experiment with parameters, invoke `python3 train.py --help` to get a list of command line arguments that modify parameters. Currently training just prints out the loss of each game episode run, without any further analysis, and the model weights are not saved at the end. These features are coming soon.

* `game.py` provides a non-tensor based implementation of the game mechanics (used for game behavior exploration and random game generation during training
* `model.py` provides the full computational model including agent and game dynamics through an entire episode
* `train.py` provides the training harness that runs many games and trains the agents
* `configs.py` provides the data structures that are passed as configuration to various modules in the computational graph as well as the default values used in training now
* `constants.py` provides constant factors that shouldn't need modification during regular running of the model