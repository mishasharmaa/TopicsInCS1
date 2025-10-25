# TopicsInCS1

## Assignment 1 - DRL for Automated Testing 
Fruit Catchers by Misha Sharma 
--- 
This game is all about catching the black basket with all the fruits falling from the sky and avoiding the black bombs that fall at the same time. Through RL, the learning agents quickly grasped how the game was supposed to function, specifically focusing on catching the fruit, avoiding bombs, using the space bar to activate the power-up, and actively striving to achieve a high score in the game. Through negative rewards and adjusted behaviour over time, the trained models understood the failure states in the game. The train agent also understood that if the basket remained on the lower half of the screen, it would be rewarded and minimize risks. 

For Fruit Catchers:
- Run 'python3 eval_agent.py' to watch AI play the game
- Run 'python3 train_agent.py' to train the game using PPO_10.
- PPO_10 is the best version the AI played the game.
- CSV files are saved in the 'models' and 'logs_csv' folders
