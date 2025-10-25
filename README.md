## Welcome
There are README's in each of the game folders that go over each of the helper files that can be run 
for each game. Each game has a training file, an evaluation file (that can be also used to visualize),
and a visualization file. These README's also go over each of the personas that the models for these games
support, as well as the arguments that their files take. 

For each of the folders, there exists four different folders that contain each of the requirements of the project.
The first folder is the log's folder. This folder contains the output of the evaluation script for its respective 
game. This folder can be changed using the logdir flag that can be seen in the respective README's. 
The second folder is the models folder that contains the models that have been trained using the training file. 
This folder can also be changed using the modeldir flag that can be seen in the respective README's.
The third folder is the src folder that contains all the code for training, evaluating, and visualizing 
the game. The environment for the game is also present in this folder as it is just a single file. 
The final folder is the tf_logs which contains the outputs of the model training and can be viewed using tensorboard.

Due to the massive amounts of difference between the games, and the number of flags that can be passed to each, there is 
no single unified environment, and the commands for each game are in the respective games README. However, if you would 
not like to read that documentation, here are the commands for running each of the best models for each game from this 
folder:

Lower Learning Rate Survival Training: python snake/src/train_snake.py --timesteps 10000000 --reward_mode survival --learning_rate 2.5e-5

Lower Learning Rate Survival Evaluation: python snake/src/snake_eval.py --reward_mode survival --model_path snake/models/ppo_snake_survival_lower_learning_rate

Lower Learning Rate Survival Visualisation: python snake/src/visualize_snake.py --reward_mode survival --model_path snake/models/ppo_snake_survival_lower_learning_rate

Lower Learning Rate Accuracy Training:  python aim_trainer/src/train_aim_trainer.py --timesteps 10000000 --learning_rate 1e-5 --reward_mode accuracy

Lower Learning Rate Accuracy Evaluation: python aim_trainer/src/eval_aim_trainer.py --reward_mode accuracy --model_path aim_trainer/models/ppo_aim_trainer_accuracy_lower_learning_rate --max_steps 10000

Lower Learning Rate Accuracy Visualization: python aim_trainer/src/visualize_aim_trainer.py --reward_mode accuracy --model_path aim_trainer/models/ppo_aim_trainer_accuracy_lower_learning_rate --max_steps 10000

## References
As talked about in the README's, the AI tools were used in the creation of some of the code including the 
rewards functions, but was not used in any of the writeup's or other parts of the project. The chats used for 
both models have been provided (for Zach's levels aim trainer and snake)

https://chatgpt.com/share/68fc25e5-68cc-8002-ad01-38ea676d63fc

https://claude.ai/share/355c87ec-85a4-4ffb-8966-1bb178c055b1

https://github.com/rajatdiptabiswas/snake-pygame

https://github.com/JacobsProjects/Python-Aim-Trainer

https://medium.com/@danushidk507/ppo-algorithm-3b33195de14a

https://gymnasium.farama.org/index.html

https://stable-baselines.readthedocs.io/en/master/guide/rl_tips.html
