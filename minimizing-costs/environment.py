import numpy as np

# Usersparameters conditions    
MONTHLY_ATMOSPHERIC_TEMPERATURE = [32.0, 30.0, 25.0, 21.0, 20.0, 18.0, 15.0, 16.0, 19.0, 22.0, 23.0, 28.0]
MIN_TEMPERATURE = -20
MAX_TEMPERATURE = 80
MIN_NUM_USERS = 10
MAX_NUM_USERS = 100
MAX_UPDATE_USERS = 5

# Data parameters conditions
MIN_RATE_DATA = 20
MAX_RATE_DATA = 300
MAX_UPDATE_DATA = 10


class Environment(object):
    """
    Environment class that:
    * Initializes Parameters and Variables
    * Updates Environment after Action
    * Method to reset environment
    * Method to retrieve last reward, current state, and if the game is over.
    """
    def __init__(
            self, 
            optimal_temperature=(18.0, 24.0),
            initial_month=1,
            initial_number_users=10,
            initial_rate_data=60,
        ):
        self.monthly_atmospheric_temperature = MONTHLY_ATMOSPHERIC_TEMPERATURE
        self.initial_month = initial_month
        self.atmospheric_temperature = self.monthly_atmospheric_temperature[initial_month]
        self.optimal_temperature = optimal_temperature
        self.min_temperature = MIN_TEMPERATURE
        self.max_temperature = MAX_TEMPERATURE
        self.min_num_users = MIN_NUM_USERS
        self.max_num_users = MAX_NUM_USERS
        self.max_update_users = MAX_UPDATE_USERS
        self.min_rate_data = MIN_RATE_DATA
        self.max_rate_data = MAX_RATE_DATA
        self.max_update_data = MAX_UPDATE_DATA
        self.initial_number_users = initial_number_users
        self.current_number_users = initial_number_users
        self.initial_rate_data = initial_rate_data
        self.current_rate_data = initial_rate_data
        self.intrinsic_temperature = self.atmospheric_temperature + 1.25 * self.current_number_users + 1.25 * self.current_rate_data
        self.temperature_ai = self.intrinsic_temperature
        self.temperature_no_ai = (self.optimal_temperature[0] + self.optimal_temperature[1]) / 2.0
        self.total_energy_ai = 0.0
        self.total_energy_no_ai = 0.0
        # Mandatory for all QLearning Model
        self.reward = 0.0
        self.game_over = 0
        self.train = 1
    
    def update_env(self, direction, energy_ai, month):
        energy_no_ai = 0
        if self.temperature_no_ai < self.optimal_temperature[0]:
            energy_no_ai = self.optimal_temperature[0] - self.temperature_no_ai
            self.temperature_no_ai = self.optimal_temperature[0]
        elif self.temperature_no_ai > self.optimal_temperature[1]:
            energy_no_ai = self.temperature_no_ai - self.optimal_temperature[1]
            self.temperature_no_ai = self.optimal_temperature[1]
    
        self.reward = energy_no_ai - energy_ai
        self.reward = 1e-3 * self.reward

        self.atmospheric_temperature = self.monthly_atmospheric_temperature[month]
        # Updating Users Variables
        self.current_number_users += np.random.randint(- self.max_update_users, self.max_update_users)
        if self.current_number_users > self.max_num_users:
            self.current_number_users = self.max_num_users
        elif self.current_number_users < self.min_num_users:
            self.current_number_users = self.min_num_users
        # Updating Data Rates Variables
        self.current_rate_data += np.random.randint(- self.max_rate_data, self.max_rate_data)
        if self.current_rate_data > self.max_rate_data:
            self.current_rate_data = self.max_rate_data
        elif self.current_rate_data < self.min_rate_data:
            self.current_rate_data = self.min_rate_data
        # Compúte the Delta of Intrinsic Temperature
        past_intrinsic_temperature = self.intrinsic_temperature
        self.intrinsic_temperature = self.atmospheric_temperature + 1.25 * self.current_number_users + 1.25 * self.current_rate_data
        delta_intrinsic_temperature =  self.intrinsic_temperature - past_intrinsic_temperature
        # Compute the Delta of Temperature caused by AI
        if direction == -1:
            delta_temperature_ai = -energy_ai
        elif direction == 1:
            delta_temperature_ai = energy_ai
        
        # Updating the new Server's Temperature with AI
        self.temperature_ai += delta_intrinsic_temperature + delta_temperature_ai
        # Updating the new Server's Temperature without AI
        self.temperature_no_ai += delta_intrinsic_temperature

        # GETTING GAME OVER
        if self.temperature_ai < self.min_temperature:
            if self.train == 1:
                self.game_over = 1
            else:
                self.temperature_ai = self.optimal_temperature[0]
                self.total_energy_ai += self.optimal_temperature[0] - self.temperature_ai

        elif self.temperature_ai > self.max_temperature:
            if self.train == 1:
                self.game_over = 1
            else:
                self.temperature_ai = self.optimal_temperature[1]
                self.total_energy_ai += self.temperature_ai - self.optimal_temperature[1] 

        # UPDATE SCORES
        self.total_energy_ai += energy_ai
        self.total_energy_no_ai += energy_no_ai
        # SCALING THE NEXT STATE
        scaled_temperature_ai = (self.temperature_ai - self.min_temperature) / (self.max_temperature - self.min_temperature)
        scaled_num_users = (self.current_number_users - self.min_num_users) / (self.max_num_users - self.min_num_users)
        scaled_rate_data = (self.current_rate_data - self.min_rate_data) / (self.max_rate_data - self.min_rate_data)
        # CREATE NEXT STATE VECTOR
        next_state = np.matrix([scaled_temperature_ai, scaled_num_users, scaled_rate_data])

        return next_state, self.reward, self.game_over

    # CREATING A METHOD THAT RESETS THE ENVIRONMENT
    def reset(self, new_month):
        self.atmospheric_temperature = self.monthly_atmospheric_temperature[new_month]
        self.initial_month = new_month
        self.current_number_users = self.initial_number_users
        self.current_rate_data = self.initial_rate_data
        self.intrinsic_temperature = self.atmospheric_temperature + 1.25 * self.current_number_users + 1.25 * self.current_rate_data
        self.temperature_ai = self.intrinsic_temperature
        self.temperature_no_ai = (self.optimal_temperature[0] + self.optimal_temperature[1]) / 2.0
        self.total_energy_ai = 0.0
        self.total_energy_no_ai = 0.0
        self.reward = 0.0
        self.game_over = 0
        self.train = 1

    # METHOD TO RETRIEVE AT ANY TIME THE CURRENT STATE, LAST REWARD, GAME OVER
    def observe(self):
        scaled_temperature_ai = (self.temperature_ai - self.min_temperature) / (self.max_temperature - self.min_temperature)
        scaled_num_users = (self.current_number_users - self.min_num_users) / (self.max_num_users - self.min_num_users)
        scaled_rate_data = (self.current_rate_data - self.min_rate_data) / (self.max_rate_data - self.min_rate_data)
        current_state = np.matrix([scaled_temperature_ai, scaled_num_users, scaled_rate_data])

        return current_state, self.reward, self.game_over
