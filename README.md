Learning to play Pong with Keras
================================

This repo has implementations of reinforcement learning (RL) algorithms in Keras to learn to play Atari Pong.
This is what I wrote in the process of learning about these RL algorithms and, it's not
well-tuned or performant. If you're looking for a reference implementation to compare against, check out
[OpenAI baselines](https://github.com/openai/baselines).

- `pong-pg.py` - Policy gradient (REINFORCE algorithm). [Defeats](https://www.youtube.com/watch?v=eYp6MeADc8I) the "computer" opponent after 400 episodes of training.
- `pong-actor-critic.py` - On-policy batch actor-critic. [Defeats](https://www.youtube.com/watch?v=rs2B6gPP49k) the "computer" opponent after 300 episodes of training.

Requirements
------------

- Python 3.6
- NumPy
- Tensorflow 1.14
- OpenAI gym


