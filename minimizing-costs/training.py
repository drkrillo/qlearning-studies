import os
import numpy as np
import random as rn
import environment
import brain
import dqn

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

# Setting parameters
epsilon = 0.3 # Exploration parameter
num_actions = 5
direction_boundary = (num_actions - 1) / 2
num_epochs = 100
max_memory = 3000
batch_size = 512
temperature_step = 1.5

env = environment.Environment(
            optimal_temperature=(18.0, 24.0),
            initial_month=1,
            initial_number_users=20,
            initial_rate_data=30,
)

brain = brain.Brain(
            lr=0.0001,
            num_actions=num_actions,
)

dqn = dqn.DQN(
            max_memory=max_memory, 
            discount=0.9
)

train = True

env.train = train
model = brain.model
if env.train:
    # TRAINING LOOP
    for epoch in range(1, num_epochs):
        # INITIALIZE VARIABLES FOR ENVIRONMENT AND TRAINING LOOP
        total_reward = 0
        loss = 0.0
        new_month = np.random.randint(0,12)
        env.reset(new_month=new_month)
        game_over = False
        current_state, _, _ = env.observe()
        timestep = 0
        # EPOCH LOOP
        while (not game_over) and (timestep <= 5 * 30 * 24 * 60):
            # PLAYING NEXT ACTION BY EXPLORATION (0.3 of times)
            if np.random.rand() < epsilon:
                action = np.random.randint(0, num_actions)
            # PLAYING NEXT ACTION BY INFERENCE/EXPLOITATION (0.7 of times)
            else:
                q_values = model.predict(current_state)
                action = np.argmax(q_values[0])
            if action - direction_boundary < 0:
                direction = -1
            else:
                direction = 1
            energy_ai = abs(action - direction_boundary) * temperature_step

            # UPDATE ENVIRONMENT AND REACH NEXT STATE
            next_state, reward, game_over = env.update_env(
                direction=direction, 
                energy_ai=energy_ai,
                month=new_month + int(timestep/(30*24*60)) 
            )
            total_reward += reward
            dqn.remember(transition=[current_state, action, reward, next_state], game_over=game_over)
            
            inputs, targets = dqn.get_batch(model=model, batch_size=batch_size)

            loss += model.train_on_batch(inputs, targets)
            timestep += 1
            current_state = next_state

        print("\n")
        print("Epoch: {:03d}/{:03d}".format(epoch, num_epochs))
        print("Total Energy spent with AI: {:.0f}".format(env.total_energy_ai))
        print("Total Energy spend without AI: {:.0f}".format(env.total_energy_no_ai))
    
    model.save("model.h5")


