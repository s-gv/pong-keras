# Copyright (c) 2019 Sagar Gubbi. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

import sys
import numpy as np
import gym

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Lambda, Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam
import tensorflow.keras.backend as K

env = gym.make('PongDeterministic-v4')

UP_ACTION = 2
DOWN_ACTION = 3
ACTIONS = [UP_ACTION, DOWN_ACTION]

# Neural net model takes the state and outputs action and value for that state
model = Sequential([
    Dense(512, activation='elu', input_shape=(2*6400,)),
    Dense(len(ACTIONS), activation='softmax'),
])

model.compile(optimizer=RMSprop(1e-4), loss='sparse_categorical_crossentropy')

gamma = 0.99

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

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward. http://karpathy.github.io/2016/05/31/rl/  """
    discounted_r = np.zeros((len(r),))
    running_add = 0
    for t in reversed(range(0, len(r))):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def train():
    reward_sums = []
    for ep in range(2000):
        Xs, ys, rewards = [], [], []
        prev_obs, obs = None, env.reset()
        for t in range(99000):
            x = np.hstack([prepro(obs), prepro(prev_obs)])
            prev_obs = obs

            action_probs = model.predict(x[None, :])
            ya = np.random.choice(len(ACTIONS), p=action_probs[0])
            action = ACTIONS[ya]

            obs, reward, done, _ = env.step(action)

            Xs.append(x)
            ys.append(ya)
            rewards.append(reward)

            #if reward != 0: print(f'Episode {ep} -- step: {t}, ya: {ya}, reward: {reward}')

            if done:
                Xs = np.array(Xs)
                ys = np.array(ys)
                discounted_rewards = discount_rewards(rewards)
                advantages = (discounted_rewards - discounted_rewards.mean()) / discounted_rewards.std()
                print(f'adv: {np.min(advantages):.2f}, {np.max(advantages):.2f}')
                
                model.fit(Xs, ys, sample_weight=advantages, epochs=1, batch_size=1024)
                
                reward_sum = sum(rewards)
                reward_sums.append(reward_sum)
                avg_reward_sum = sum(reward_sums[-50:]) / len(reward_sums[-50:])
                
                print(f'Episode {ep} -- reward_sum: {reward_sum}, avg_reward_sum: {avg_reward_sum}\n')

                if ep % 20 == 0:
                    model.save_weights('params/model3.h5')
                break

def test():
    global env
    env = gym.wrappers.Monitor(env, './tmp', video_callable=lambda ep_id: True, force=True)

    model.load_weights('params/model3.h5')
    
    reward_sum = 0
    prev_obs, obs = None, env.reset()
    for t in range(99000):
        x = np.hstack([prepro(obs), prepro(prev_obs)])
        prev_obs = obs

        action_probs = model.predict(x[None, :])
        #ya = np.argmax(action_probs[0])
        ya = np.random.choice(len(ACTIONS), p=action_probs[0])
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

