Learning to play Pong
=====================

This repo has implementations of reinforcement learning (RL) algorithms in Keras to learn to play Atari Pong.
This is what I wrote in the process of learning about these RL algorithms, so it's not
well-tuned or performant. If you're looking for a reference implementation to compare against, check out
[OpenAI baselines](https://github.com/openai/baselines).

- `pong-ppo.py` - PPO. [Defeats](https://www.youtube.com/watch?v=gf9rRNBw9QE) the "computer" opponent after 300 episodes of training.
- `pong-pg.py` - Policy gradient (REINFORCE algorithm). [Defeats](https://www.youtube.com/watch?v=eYp6MeADc8I) the "computer" opponent after 400 episodes of training.
- `pong-actor-critic.py` - On-policy batch actor-critic. [Defeats](https://www.youtube.com/watch?v=rs2B6gPP49k) the "computer" opponent after 300 episodes of training.
- `pong-ddqn-batch.py` - Off-policy double Q learning. [Defeats](https://www.youtube.com/watch?v=_VnRkKAcnFI) the "computer" opponent after 2000 episodes of training.
- `pong-ddqn-per.py` - Off-policy double Q learning with prioritized experience replay. [Defeats](https://www.youtube.com/watch?v=rb9QmFbY7kE) the "computer" opponent after 1000 episodes of training.

Requirements
------------

- Python 3.6
- NumPy
- Tensorflow 1.14
- OpenAI gym


