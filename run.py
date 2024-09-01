from dqn_agent import DQNAgent
from tetris import Tetris
from datetime import datetime
from statistics import mean
from logs import CustomTensorBoard
from tqdm import tqdm
        

# Run dqn with Tetris
def dqn():
    env = Tetris()
    episodes = 3000 # total number of episodes
    max_steps = None # max number of steps per game (None for infinite)
    epsilon_stop_episode = 2000 # at what episode the random exploration stops
    mem_size = 1000 # maximum number of steps stored by the agent
    discount = 0.95 # discount in the Q-learning formula (see DQNAgent)
    batch_size = 128 # number of actions to consider in each training
    epochs = 1 # number of epochs per training
    render_every = 50 # renders the gameplay every x episodes
    render_delay = None # delay added to render each frame (None for no delay)
    log_every = 50 # logs the current stats every x episodes
    replay_start_size = 1000 # minimum steps stored in the agent required to start training
    train_every = 1 # train every x episodes
    n_neurons = [32, 32, 32] # number of neurons for each activation layer
    activations = ['relu', 'relu', 'relu', 'linear'] # activation layers
    save_best_model = True # saves the best model so far at "best.keras"

    agent = DQNAgent(env.get_state_size(),
                     n_neurons=n_neurons, activations=activations,
                     epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size,
                     discount=discount, replay_start_size=replay_start_size)

    log_dir = f'logs/tetris-nn={str(n_neurons)}-mem={mem_size}-bs={batch_size}-e={epochs}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    log = CustomTensorBoard(log_dir=log_dir)

    scores = []
    best_score = 0

    for episode in tqdm(range(episodes)):
        current_state = env.reset()
        done = False
        steps = 0

        if render_every and episode % render_every == 0:
            render = True
        else:
            render = False

        # Game
        while not done and (not max_steps or steps < max_steps):
            # state -> action
            next_states = {tuple(v):k for k, v in env.get_next_states().items()}
            best_state = agent.best_state(next_states.keys())
            best_action = next_states[best_state]

            reward, done = env.play(best_action[0], best_action[1], render=render,
                                    render_delay=render_delay)
            
            agent.add_to_memory(current_state, best_state, reward, done)
            current_state = best_state
            steps += 1

        scores.append(env.get_game_score())

        # Train
        if episode % train_every == 0:
            agent.train(batch_size=batch_size, epochs=epochs)

        # Logs
        if log_every and episode and episode % log_every == 0:
            avg_score = mean(scores[-log_every:])
            min_score = min(scores[-log_every:])
            max_score = max(scores[-log_every:])

            log.log(episode, avg_score=avg_score, min_score=min_score,
                    max_score=max_score)

        # Save model
        if save_best_model and env.get_game_score() > best_score:
            print(f'Saving a new best model (score={env.get_game_score()}, episode={episode})')
            best_score = env.get_game_score()
            agent.save_model("best.keras")


if __name__ == "__main__":
    dqn()
