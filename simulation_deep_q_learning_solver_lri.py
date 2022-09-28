import tdqc
import numpy as np
from tdqc.solver.deep_q_learning import DeepQLearning, DQLWithReplayMemory
from tdqc.numerics.ed.models_ed import State
from tdqc.numerics.ed.models_ed import xxz_model
from tdqc.numerics.deep_q_learning.parameters_lri import parameters, parameters_replay_memory
solver = DQLWithReplayMemory()
solver.load_settings(settings=parameters)
solver.load_seetings_replay_memory(**parameters_replay_memory)

solver.solve()
