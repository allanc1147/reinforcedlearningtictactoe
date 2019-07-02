import tensorflow as tf
import numpy as np
from collections import deque
from tictactoe.deep_q_network import DeepQNetwork
from tictactoe.game import Game

# initialize game env
env = Game()

# initialize tensorflow
sess1 = tf.Session()
sess2 = tf.Session()
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9)
writer1 = tf.train.SummaryWriter("logs/value_network1", sess1.graph)
writer2 = tf.train.SummaryWriter("logs/value_network2", sess2.graph)

# prepare custom tensorboard summaries
episode_reward1 = tf.Variable(0.)
episode_reward2 = tf.Variable(0.)
tf.scalar_summary("Last 100 Episodes Average Episode Reward for player 1", episode_reward1)
tf.scalar_summary("Last 100 Episodes Average Episode Reward for player 2", episode_reward2)
summary_vars1 = [episode_reward1]
summary_vars2 = [episode_reward2]

summary_placeholders1 = [tf.placeholder("float") for i in range(len(summary_vars1))]
summary_placeholders2 = [tf.placeholder("float") for i in range(len(summary_vars2))]
summary_ops1 = [summary_vars1[i].assign(summary_placeholders1[i]) for i in range(len(summary_vars1))]
summary_ops2 = [summary_vars2[i].assign(summary_placeholders2[i]) for i in range(len(summary_vars2))]

# define policy neural network
state_dim = 9
num_actions = 9


def value_network(states):
    W1 = tf.get_variable("W1", [state_dim, 256],
                         initializer=tf.random_normal_initializer(stddev=0.1))
    b1 = tf.get_variable("b1", [256],
                         initializer=tf.constant_initializer(0))
    h1 = tf.nn.relu(tf.matmul(states, W1) + b1)

    W2 = tf.get_variable("W2", [256, 64],
                         initializer=tf.random_normal_initializer(stddev=0.1))
    b2 = tf.get_variable("b2", [64],
                         initializer=tf.constant_initializer(0))
    h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)

    Wo = tf.get_variable("Wo", [64, num_actions],
                         initializer=tf.random_normal_initializer(stddev=0.1))
    bo = tf.get_variable("bo", [num_actions],
                         initializer=tf.constant_initializer(0))

    p = tf.matmul(h2, Wo) + bo
    return p


summaries = tf.merge_all_summaries()
q_network1 = DeepQNetwork(sess1,
                          optimizer,
                          value_network,
                          state_dim,
                          num_actions,
                          init_exp=0.6,  # initial exploration prob
                          final_exp=0.1,  # final exploration prob
                          anneal_steps=120000,  # N steps for annealing exploration
                          discount_factor=0.8)  # no need for discounting
q_network2 = DeepQNetwork(sess2,
                          optimizer,
                          value_network,
                          state_dim,
                          num_actions,
                          init_exp=0.6,  # initial exploration prob
                          final_exp=0.1,  # final exploration prob
                          anneal_steps=120000,  # N steps for annealing exploration
                          discount_factor=0.8)  # no need for discounting

# load checkpoint if there is any
saver = tf.train.Saver()
checkpoint1 = tf.train.get_checkpoint_state("model1")
checkpoint2 = tf.train.get_checkpoint_state("model2")
if checkpoint1 and checkpoint1.model_checkpoint_path:
    saver.restore(sess1, checkpoint1.model_checkpoint_path)
    print("successfully loaded checkpoint1")
if checkpoint2 and checkpoint2.model_checkpoint_path:
    saver.restore(sess2, checkpoint2.model_checkpoint_path)
    print("successfully loaded checkpoint2")

# how many episodes to train
training_episodes = 200000

# store episodes history
episode_history1 = deque(maxlen=100)
episode_history2 = deque(maxlen=100)

lost1 = 0
draw1 = 0
won1 = 0
cheated1 = 0

lost2 = 0
draw2 = 0
won2 = 0
cheated2 = 0

# start training
reward1 = 0.0
reward2 = 0.0
for i_episode in range(training_episodes):
    state = np.array(env.reset())
    for t in range(20):
        # player 1
        action1 = q_network1.eGreedyAction(state[np.newaxis, :])
        next_state, reward1, done1, redo1 = env.step(action1, 'x')

        state1cpy = state.copy()
        next_state1cpy = next_state.copy()

        if done1:
            q_network1.storeExperience(state1cpy, action1, reward1 - reward2, next_state1cpy, done1)
            q_network1.updateModel()
            q_network2.storeExperience(state2cpy, action2, reward1 - reward2, next_state2cpy, done2)
            q_network2.updateModel()

            if reward1 == -10:
                cheated1 += 1
            elif reward1 == 100:
                won1 += 1
                lost2 += 1
            elif reward1 == 0:
                draw1 += 1
                draw2 += 1
            episode_history1.append(reward1 - reward2)
            episode_history2.append(reward1 - reward2)
            break

        # since we know reward from player 1, we can update model
        q_network2.storeExperience(state2cpy, action2, reward1 - reward2, next_state2cpy, done2)
        q_network2.updateModel()

        state = np.array(next_state)

        # player 2
        action2 = q_network2.eGreedyAction(state[np.newaxis, :])
        next_state, reward2, done2, redo2 = env.step(action2, 'o')

        state2cpy = state.copy()
        next_state2cpy = next_state.copy()

        if done2:
            q_network1.storeExperience(state1cpy, action1, reward1 - reward2, next_state1cpy, done1)
            q_network1.updateModel()
            q_network2.storeExperience(state2cpy, action2, reward1 - reward2, next_state2cpy, done2)
            q_network2.updateModel()

            if reward2 == -10:
                cheated2 += 1
            elif reward2 == 100:
                won2 += 1
                lost1 += 1
            elif reward2 == 0:
                draw1 += 1
                draw2 += 1
            episode_history1.append(reward1 - reward2)
            episode_history2.append(reward1 - reward2)
            break

        # since we know reward from player 2, we can update model
        q_network1.storeExperience(state1cpy, action1, reward1 - reward2, next_state1cpy, done1)
        q_network1.updateModel()

        state = np.array(next_state)

    # print status every 100 episodes
    if i_episode % 100 == 0:
        mean_rewards1 = np.mean(episode_history1)
        print("Episode {}".format(i_episode))
        print("Reward for this episode for p1: {}".format(reward1))
        print("Average reward for player 1 for last 100 episodes: {}".format(mean_rewards1))
        print("cheated1:" + str(cheated1))
        print("lost1:" + str(lost1))
        print("won1:" + str(won1))
        print("draw1:" + str(draw1))
        mean_rewards2 = np.mean(episode_history2)
        print("Reward for this episode for p2: {}".format(reward2))
        print("Average reward for player 2 for last 100 episodes: {}".format(mean_rewards2))
        print("cheated2:" + str(cheated2))
        print("lost2:" + str(lost2))
        print("won2:" + str(won2))
        print("draw2:" + str(draw2))
        # update tensorboard
        sess1.run(summary_ops1[0], feed_dict={summary_placeholders1[0]: float(mean_rewards1)})
        result1 = sess1.run(summaries)
        writer1.add_summary(result1, i_episode)
        sess2.run(summary_ops2[0], feed_dict={summary_placeholders2[0]: float(mean_rewards2)})
        result2 = sess2.run(summaries)
        writer2.add_summary(result2, i_episode)

        lost1 = 0
        draw1 = 0
        won1 = 0
        cheated1 = 0
        lost2 = 0
        draw2 = 0
        won2 = 0
        cheated2 = 0

        # save checkpoint
        saver.save(sess1, "model1/saved_network")
        saver.save(sess2, "model2/saved_network")
