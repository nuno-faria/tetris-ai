#!/usr/bin/python3

from dqn_agent import DQNAgent
from zentris import Zentris
from datetime import datetime
from statistics import mean, median
import random
from logs import CustomTensorBoard
from tqdm import trange
import argparse
import tensorflow.summary
from os import system
from sys import argv

# Run dqn with Zentris
def dqn(args):
    env = Zentris()

    episodes = args.nbepisodes
    max_steps = args.maxsteps
    epsilon = args.epsilon[0]
    epsilon_min = args.epsilon[1]
    epsilon_stop_episode = args.epsdecrease
    force_fail = args.forcefail
    score_when_fail = -args.scorefail
    discount = args.discount
    batch_size = args.batch
    epochs = args.epochs
    optimizer = args.optimizer
    mem_size = 100000

    nb_channels_small_filter = args.channels[0]
    nb_channels_big_filter   = args.channels[1]
    nb_dense_neurons         = args.dense

    replay_start_size = 1024
    render_every = 5
    log_every = 50
    save_every = 200
    train_every = 1
    render_delay = None
    loss = 'mse'

    agent = DQNAgent(env.get_state_info()[0],
                     nb_channels_small_filter=nb_channels_small_filter, nb_channels_big_filter=nb_channels_big_filter, nb_dense_neurons=nb_dense_neurons,
                     epsilon=epsilon, epsilon_min=epsilon_min, epsilon_stop_episode=epsilon_stop_episode,
                     mem_size=mem_size, discount=discount, replay_start_size=replay_start_size,
                     optimizer=optimizer, loss=loss)

    state_descr = env.get_state_info()[1]
    log_dir = f'logs/{datetime.now().strftime("%m%d-%H%M")}-{state_descr}-dsc{discount}-eps{epsilon}-{epsilon_min}-bsz{batch_size}-ep{epochs}-nn{nb_channels_small_filter}-{nb_channels_big_filter}-{str(nb_dense_neurons)}-ff{force_fail}_{score_when_fail}_{optimizer}'
    log = CustomTensorBoard(log_dir=log_dir)
    system('cp -v dqn_agent.py logs.py runZentris.py zentris.py "' + log_dir+'/"')
    system('echo ' + str(argv) + ' >> "' + log_dir+'/runZentris.py"')
    print()
    print(args)
    print(log_dir)

    scores_all     = []
    scores_no_rand = []
    steps_all      = []
    steps_no_rand  = []
    losses         = []
    failqs         = []

    for episode in trange(episodes+1, ncols=80):
        current_state = env.reset()
        done = False
        steps = 0
        render = (render_every and (episode+1) % render_every == 0)

        # Game
        while not done and (not max_steps or steps < (max_steps if not render else 3*max_steps)):
            next_states = env.get_next_states()
            if len(next_states) == 0:
                done = True
            else:
                best_action, best_state = agent.find_best_state(next_states, allow_random=not(render))
                reward = env.play(best_action, render=False)
                
                agent.add_to_memory(current_state, next_states[best_action], reward, done)
                current_state = next_states[best_action]
            steps += 1

        if done:
            agent.add_to_memory(current_state, current_state, score_when_fail, done)
        if render:
            # for p in env.pieces:
            #     p.print()
            #     print()
            #print('FINAL SCORE =', env.get_game_score())
            #print(log_dir)
            scores_no_rand.append(env.get_game_score())
            steps_no_rand.append(steps)
        scores_all.append(env.get_game_score())
        steps_all.append(steps)

        # Train
        if episode % train_every == 0:
            loss, failq_median = agent.train(batch_size=batch_size, epochs=epochs, force_fail=force_fail)
            losses.append(loss         if loss         else score_when_fail)
            failqs.append(failq_median if failq_median else 0)

        # Logs
        if log_every and episode and (episode % log_every == 0 or episode in [log_every//2,log_every//4,log_every//8]):
            if scores_no_rand:
                eval_score_median = median(scores_no_rand[-log_every//render_every:])
                eval_score_min    = min   (scores_no_rand[-log_every//render_every:])
                eval_score_max    = max   (scores_no_rand[-log_every//render_every:])
                eval_steps_median = median(steps_no_rand [-log_every//render_every:])
                eval_steps_min    = min   (steps_no_rand [-log_every//render_every:])
                eval_steps_max    = max   (steps_no_rand [-log_every//render_every:])
            else:
                eval_score_median, eval_score_min, eval_score_max = 0, 0, 0
                eval_steps_median, eval_steps_min, eval_steps_max = 0, 0, 0
            score_median      = median(scores_all[-log_every:])
            steps_median      = median(steps_all[-log_every:])
            loss_median  = median(losses[-log_every//train_every:]) if losses else -1
            failq_median = median(failqs[-log_every//train_every:]) if failqs else -1

            log.log(episode,
                eval_score_median=eval_score_median, eval_score_min=eval_score_min, eval_score_max=eval_score_max,
                score_median=score_median,
                eval_steps_median=eval_steps_median, eval_steps_min=eval_steps_min, eval_steps_max=eval_steps_max,
                steps_median=steps_median,
                loss_median=loss_median, failq_median=failq_median,
                epsilon=agent.epsilon,
                )

        if save_every and episode and episode % save_every == 0:
            agent.save(filename=log_dir + '/' + f'model_{datetime.now().strftime("%m%d-%H%M")}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nbepisodes'  , '-n', action='store' , default=2000  , type=int,    help='nb of episodes to run')
    parser.add_argument('--maxsteps'    , '-m', action='store' , default=25    , type=int,    help='max steps per episode')
    parser.add_argument('--epsilon'     , '-E', action='store' , default=[0.1,0.01], nargs=2, type=float, help='epsilon value (percentage of random moves) at beginning and at the end')
    parser.add_argument('--epsdecrease' , '-e', action='store' , default=1000  , type=int,    help='nb of episodes to decrease epsilon value')
    parser.add_argument('--optimizer'   , '-o', action='store' , default='adam', type=str,    help='optimizer and params (like adam-3-7 for lr=1e-3 and eps=1e-7)')
    parser.add_argument('--forcefail'   , '-f', action='store' , default=10    , type=int,    help='min percentage of failures for training')
    parser.add_argument('--scorefail'   , '-F', action='store' , default=10000 , type=int,    help='score to give when failure (no negative sign)')
    parser.add_argument('--discount'    , '-D', action='store' , default=0.9   , type=float,  help='discount value btw 0-1 (closer to 1=long-term view)')
    parser.add_argument('--batch'       , '-b', action='store' , default=1024  , type=int,    help='batch size')
    parser.add_argument('--epochs'      , '-p', action='store' , default=5     , type=int,    help='nb of epochs when training')
    parser.add_argument('--channels'    , '-c', action='store' , default=[6,8]  , nargs=2, type=int, help='(CNN archi) nb of channels for small filters and big filters')
    parser.add_argument('--dense'       , '-d', action='store' , default=[6,6,3], nargs='*', type=int, help='(CNN archi) nb of neurons for dense layers at the end channels')
    args = parser.parse_args()

    dqn(args)
