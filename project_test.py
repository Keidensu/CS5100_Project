import numpy as np

class TaxiEnv:
    def __init__(self):
        # Customize the environment settings
        self.grid_size = 5
        self.num_states = self.grid_size ** 2
        self.num_actions = 6  # 4 directions + pick-up + drop-off
        self.state = self.reset()  # Use reset method to initialize state
        self.passenger_loc = (1, 2)  # Passenger initial location
        self.destination_loc = (3, 4)  # Destination initial location
        self.q_table = np.zeros((self.num_states, self.num_actions))
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 1.0
        self.min_exploration_rate = 0.01
        self.exploration_decay_rate = 0.99

    def reset(self):
        """
        Reset the environment to the initial state and return it.
        Can be extended to reset passenger and destination locations.
        """
        self.state = 0  # Resets only the taxi's location for now
        return self.state

    def get_possible_actions(self):
        """
        Return a list of possible actions.
        """
        return [0, 1, 2, 3, 4, 5]  # 0: up, 1: down, 2: left, 3: right, 4: pick-up, 5: drop-off

    def step(self, action):
        """
        Take an action and return the next state, reward, and done flag.
        """
        done = False  # Indicates whether the episode has ended
        if action < 4:  # Movement actions
            self.move(action)
            reward = -1  # Standard movement cost
        elif action == 4:  # pick-up action
            reward, done = self.pickup()
        elif action == 5:  # drop-off action
            reward, done = self.dropoff()
        else:
            raise ValueError("Invalid action")

        return self.state, reward, done

    def move(self, action):
        """
        Update the state based on the movement action.
        """
        if action == 0 and self.state >= self.grid_size:  # Move up
            self.state -= self.grid_size
        elif action == 1 and self.state < self.num_states - self.grid_size:  # Move down
            self.state += self.grid_size
        elif action == 2 and self.state % self.grid_size != 0:  # Move left
            self.state -= 1
        elif action == 3 and (self.state + 1) % self.grid_size != 0:  # Move right
            self.state += 1

    def pickup(self):
        """
        Handle the pick-up action and return the reward and done flag.
        """
        if self.state == self.passenger_loc[0] * self.grid_size + self.passenger_loc[1]:
            self.passenger_loc = (-1, -1)  # Passenger is now on-board
            return 10, False  # Reward for successful pick-up
        else:
            return -10, False  # Penalty for incorrect pick-up

    def dropoff(self):
        """
        Handle the drop-off action and return the reward and done flag.
        """
        if self.state == self.destination_loc[0] * self.grid_size + self.destination_loc[1]:
            return 20, True  # Reward for successful drop-off and episode ends
        else:
            return -20, False  # Penalty for incorrect drop-off

    def q_learning(self, num_episodes):
        for episode in range(num_episodes):
            state = self.reset()  # Reset the environment at the start of each episode
            total_reward = 0
            done = False
            steps = 0  # Track the number of steps taken in the episode

            while not done:
                # Choose an action based on the exploration-exploitation strategy
                action = np.random.choice(self.get_possible_actions()) if np.random.rand() < self.exploration_rate \
                    else np.argmax(self.q_table[state])

                next_state, reward, done = self.step(action)
                steps += 1  # Increment steps for each action taken

                # Update Q-value using the Q-learning formula
                self.q_table[state, action] = (1 - self.learning_rate) * self.q_table[state, action] + \
                                              self.learning_rate * (reward + self.discount_factor * np.max(self.q_table[next_state]))

                total_reward += reward
                state = next_state

            # Adjust the exploration rate
            self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay_rate)

            # Logging for every 100 episodes
            if episode % 100 == 0:
                print(f"Episode {episode}: Total Reward = {total_reward}, Steps = {steps}, Final Exploration Rate = {self.exploration_rate:.4f}")

        print("Training finished.")
    
    
    def test_agent(self):
        state = self.reset()  # Start from the initial state for testing
        total_reward = 0
        done = False
        steps = 0  # Track the number of steps during testing

        print("\nTesting trained agent...")
        while not done:
            action = np.argmax(self.q_table[state])
            state, reward, done = self.step(action)
            total_reward += reward
            steps += 1
            print(f"Step {steps}: State = {state}, Action = {action}, Reward = {reward}")

        print(f"Test Total Reward: {total_reward}, Total Steps: {steps}")

# Main execution
if __name__ == "__main__":
    env = TaxiEnv()
    num_episodes = 1000
    env.q_learning(num_episodes)
    env.test_agent()  # Call the enhanced testing method

    # After training, test the trained agent
    state = env.reset()  # Start from the initial state for testing
    total_reward = 0
    done = False

    print("\nTesting trained agent...")
    while not done:
        # Choose the best action from Q-table
        action = np.argmax(env.q_table[state])
        state, reward, done = env.step(action)
        total_reward += reward

    print(f"Test Total Reward: {total_reward}")
