# TopicsInCS1

## Assignment 1 - DRL for Automated Testing 
Fruit Catchers by Misha Sharma 
--- 
This game is all about catching the black basket with all the fruits falling from the sky and avoiding the black bombs that fall at the same time. Through RL, the learning agents quickly grasped how the game was supposed to function, specifically focusing on catching the fruit, avoiding bombs, using the space bar to activate the power-up, and actively striving to achieve a high score in the game. Through negative rewards and adjusted behaviour over time, the trained models understood the failure states in the game. The train agent also understood that if the basket remained on the lower half of the screen, it would be rewarded and minimize risks. 

### Rewards

Positive rewards were given when the basket caught the fruit and avoided bombs. Penalties were given for bomb collisions and missed fruit, and this encouraged better timing. Small alignment rewards were also given if the basket positioned itself under the fruit, which helped make the game run smoothly. 

### Persona Trade-offs

I had a "survivor" and a "collector" reward persona tested. The survivor persona prioritized staying alive, whereas the collector persona prioritized aggressive play. The survivor persona demonstrated more stability when going through testing. The collector persona showed higher reward variance, but more aggressive behaviour. 

## To Run:
- Run 'python3 eval_agent.py ppo_10' for ppo model
- Run 'python3 eval_agent.py a2c' for a2c model
- Run 'python3 eval_agent.py lr5e5' for learning rate model
- Run 'python3 train_agent.py' and add either 'ppo' 'a2c' or 'lr5e5' to train either model
- Run 'python3 plot_performance.py' to plot graph
