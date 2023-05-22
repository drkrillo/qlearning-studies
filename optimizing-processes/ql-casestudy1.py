import numpy as np

starting_location = 'E'
intermediate_locations = ['D', 'K']
ending_location = 'G'
rewards = {
    #'K': 0.5,
}

gamma = 0.75
alpha = 0.9
training_iterations=5000

location_to_state = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
    'H': 7,
    'I': 8,
    'J': 9,
    'K': 10,
    'L': 11,
}

state_to_location = {
    state: location for location, state in location_to_state.items()
}

actions = [x for x in range(12)]

R = np.array([
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # A
    [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0], # B
    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # C
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # D
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], # E
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], # F
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0], # G
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1], # H
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], # I
    [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0], # J
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], # K
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0], # L
    ])

Q = np.zeros([12,12])


def set_initial_conditions(
        R,
        ending_location,
        reward_amount=1000,
        **rewards
    ):
    """
    Use 'A-L': .0-1. in kwargs.
    It will set that key location reward to value*reward_amount
    e.g.
    'A': 0.01 with reward_amount=1000 will set:
    0.01*1000=10 to that location reward priority.
    """
    ending_state = location_to_state[ending_location]
    R[ending_state, ending_state] = reward_amount
    #for location, reward_rate in rewards.items():
    #    state = location_to_state[location]
    #    R[:, state] = [reward_rate*reward_amount if x != 0 else 0 for x in R[:, state]]

    return R


def ql_training(
        Q,
        R,
        alpha=0.9,
        gamma=0.75,
        iterations=1000,
    ):
    """
    Takes Q and R matrices, and performs Q Learning algorithm.
    Returns tuned Q and R matrices.
    """
    for _ in range(iterations):

        current_state = np.random.randint(0,12)
        playable_actions = []
        for j in range(12):
            if R[current_state, j] > 0:
                playable_actions.append(j)
        
        next_state = np.random.choice(playable_actions)

        TD = R[current_state, next_state] + gamma * Q[next_state, np.argmax(Q[next_state,])] - Q[current_state, next_state]

        Q[current_state, next_state] += alpha * TD

    return Q, R


def route(
        Q,
        starting_location, 
        ending_location,
    ):
    route = []
    next_location = starting_location
    while next_location != ending_location:
        starting_state = location_to_state[next_location]
        next_state = np.argmax(Q[starting_state,])
        next_location = state_to_location[next_state]
        route.append(next_location)
        
    return route


def optimal_route(Q, R, starting_location, ending_location, intermediate_locations):
    
    optimal_route = [starting_location]
    intermediate_locations.append(ending_location)

    for location in intermediate_locations:
        R_tmp = np.copy(R)
        Q_tmp = np.copy(Q)

        R_new = set_initial_conditions(R_tmp, ending_location=location)
        Q_new, R_new = ql_training(Q=Q_tmp, R=R_new, iterations=training_iterations)
        intermediate_route = route(Q_new, starting_location, location)
        optimal_route += intermediate_route

        starting_location = location

    return optimal_route


route = optimal_route(
    Q,
    R,
    starting_location=starting_location,
    ending_location=ending_location,
    intermediate_locations=intermediate_locations,
    )

print("ROUTE:")
print(route)
