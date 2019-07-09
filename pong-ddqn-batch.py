# Copyright (c) 2019 Sagar Gubbi. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

import sys
import numpy as np
import gym
from collections import deque
import random

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Lambda, Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam
import tensorflow.keras.backend as K

env = gym.make('PongDeterministic-v4')

#NO_ACTION = 0
UP_ACTION = 2
DOWN_ACTION = 3
ACTIONS = [UP_ACTION, DOWN_ACTION]

# Neural net model takes the state and outputs action and value for that state
net = Sequential([
    Dense(512, activation='relu', input_shape=(2*6400,)),
    Dense(128, activation='relu'),
    Dense(len(ACTIONS)),
])

net.compile(optimizer=Adam(1e-4), loss=tf.keras.losses.Huber())
target_net = tf.keras.models.clone_model(net)


# preprocess frames
def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector. http://karpathy.github.io/2016/05/31/rl/ """
    if I is None: return np.zeros((6400,))
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

def train():
    memory = deque(maxlen=64000)
    eps = 1.0
    gamma = 0.99

    reward_sums = []
    for ep in range(6000):
        prev_obs, obs = None, env.reset()
        reward_sum = 0
        for t in range(99000):
            x = np.hstack([prepro(obs), prepro(prev_obs)])
            prev_obs = obs

            if np.random.random() < eps:
                ya = np.random.choice(len(ACTIONS))
            else:
                qvals = net.predict(x[None, :])
                ya = np.argmax(qvals[0])
            action = ACTIONS[ya]

            obs, reward, done, _ = env.step(action)

            round_done = (reward != 0)
            memory.append((x, ya, reward, round_done))
            reward_sum += reward

            #if round_done: print(f'Episode {ep}, t {t}, eps {eps} -- reward: {reward}')

            if done:
                if ep % 3 == 0:
                    for k in range(2):
                        sample_idxs = random.sample(range(len(memory)-1), min(len(memory)-1, 16384))
                        
                        Xs = np.array([memory[idx][0] for idx in sample_idxs])
                        Xs_ = np.array([memory[idx+1][0] for idx in sample_idxs])
                        yas = np.array([memory[idx][1] for idx in sample_idxs])
                        rewards = np.array([memory[idx][2] for idx in sample_idxs])
                        dones = [memory[idx][3] for idx in sample_idxs]

                        Qs = net.predict(Xs)
                        Qs_ = net.predict(Xs_)
                        target_Qs_ = target_net.predict(Xs_)

                        for i in range(len(sample_idxs)):
                            ya = yas[i]
                            if dones[i]:
                                Qs[i][ya] = rewards[i]
                            else:
                                Qs[i][ya] = rewards[i] + gamma * target_Qs_[i][np.argmax(Qs_[i])]

                        net.fit(Xs, Qs, epochs=1, batch_size=64)
                
                reward_sums.append(reward_sum)
                avg_reward_sum = sum(reward_sums[-50:]) / len(reward_sums[-50:])
                print(f'Episode {ep}, t {t}, eps {eps:.2f} -- reward_sum: {reward_sum}, avg_reward_sum: {avg_reward_sum}')
                
                eps = max(0.1, 1.1 - ep/250.0)
                
                if ep % 6 == 0:
                    target_net.set_weights(net.get_weights())
                    target_net.save('params/target_net.h5')
                
                break

def test():
    global env
    env = gym.wrappers.Monitor(env, './tmp', video_callable=lambda ep_id: True, force=True)

    target_net.load_weights('params/target_net.h5')
    
    reward_sum = 0
    prev_obs, obs = None, env.reset()
    for t in range(99000):
        x = np.hstack([prepro(obs), prepro(prev_obs)])
        prev_obs = obs

        qvals = target_net.predict(x[None, :])
        ya = np.argmax(qvals[0])
        action = ACTIONS[ya]

        obs, reward, done, _ = env.step(action)
        reward_sum += reward

        if reward != 0:
            print(f't: {t} -- reward: {reward}')

        if done:
            print(f't: {t} -- reward_sum: {reward_sum}')
            break

def main():
    if len(sys.argv) >= 2 and sys.argv[1] == 'test':
        test()
    else:
        train()

if __name__ == '__main__':
    main()

