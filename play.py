#!/usr/bin/python3

from zentris import Zentris
from datetime import datetime
from statistics import mean, median
import random
from tqdm import tqdm
        

def play():
    env = Zentris()
    episodes = 100
    scores = []

    for episode in tqdm(range(episodes)):
        current_state = env.reset()
        
        # Game
        done = False
        while not done:
            next_actions_states = env.get_next_states()
            if len(next_actions_states) == 0:
                done = True
            else:
                best_action = random.choice(list(next_actions_states.keys()))
                best_state = next_actions_states[best_action]

                _, _ = env.play(best_action[0], best_action[1], render=True, render_delay=0)
                current_state = best_state

        scores.append(env.get_game_score())


if __name__ == "__main__":
    play()
