import sys

if len(sys.argv) < 2:
    exit("Missing model file")

from dqn_agent import DQNAgent
from tetris import Tetris

env = Tetris()
agent = DQNAgent(env.get_state_size(), modelFile=sys.argv[1])
done = False

while not done:
    next_states = {tuple(v): k for k, v in env.get_next_states().items()}
    best_state = agent.best_state(next_states.keys())
    best_action = next_states[best_state]
    reward, done = env.play(best_action[0], best_action[1], render=True)
