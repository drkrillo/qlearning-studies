import os
import numpy as np
import random as rn
import environment
from keras.models import load_model

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

# Setting parameters
num_actions = 5
direction_boundary = (num_actions - 1) / 2
temperature_step = 1.5

env = environment.Environment(
            optimal_temperature=(18.0, 24.0),
            initial_month=1,
            initial_number_users=20,
            initial_rate_data=30,
)

train = False # Inference Mode
model = load_model("model.h5")

# SIMULATE 12 MONTHS
current_state, _, _ = env.observe()
for timestep in range(0, 12 * 30 * 24 * 60):
    q_values = model.predict(current_state)
    action = np.argmax(q_values[0])
    if action - direction_boundary < 0:
        direction = -1
    else:
        direction = 1
    energy_ai = abs(action - direction_boundary) * temperature_step
    next_state, reward, game_over = env.update_env(
        direction=direction, 
        energy_ai=energy_ai,
        month=int(timestep/(30*24*60)) 
    )
    current_state = next_state
    print(f"{timestep} out of {12*30*24*60}")

print("\n")
print("Total Energy spent with AI: {:.0f}".format(env.total_energy_ai))
print("Total Energy spend without AI: {:.0f}".format(env.total_energy_no_ai))
print("ENERGY SAVED: {:.0f}%".format((env.total_energy_no_ai - env.total_energy_ai) / env.total_energy_no_ai * 100))
