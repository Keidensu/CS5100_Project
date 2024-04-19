import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import pygame

class TaxiEnv:
    def __init__(self):
        # Customize the environment settings
        self.grid_size = 5
        self.num_states = self.grid_size ** 2
        self.cell_size = 100
        self.num_actions = 6  # 4 directions + pick-up + drop-off
        self.state = self.reset()  # Use reset method to initialize state
        self.passenger_loc = (1, 2)  # Passenger initial location
        self.destination_loc = (3, 4)  # Destination initial location
        self.q_table_pre_pickup = np.zeros((self.num_states, self.num_actions))  # Q-table for before pickup
        self.q_table_post_pickup = np.zeros((self.num_states, self.num_actions))  # Q-table for after pickup
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 1.0
        self.min_exploration_rate = 0.01
        self.exploration_decay_rate = 0.99
        self.in_transit = False
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.grid_size * self.cell_size, self.grid_size * self.cell_size))
        pygame.display.set_caption('Taxi Environment')
    def draw_grid(self,delay):
        self.screen.fill((255, 255, 255))  # white background

        # Draw grid lines
        for x in range(0, self.grid_size * self.cell_size, self.cell_size):
            pygame.draw.line(self.screen, (0, 0, 0), (x, 0), (x, self.grid_size * self.cell_size))
        for y in range(0, self.grid_size * self.cell_size, self.cell_size):
            pygame.draw.line(self.screen, (0, 0, 0), (0, y), (self.grid_size * self.cell_size, y))
        
        obstacle_color = (127, 127, 127)
        for obstacle in self.obstacles:
            pygame.draw.rect(self.screen, obstacle_color, (
            (obstacle % self.grid_size) * self.cell_size, (obstacle // self.grid_size) * self.cell_size,
            self.cell_size, self.cell_size))
        # Draw taxi
        pygame.draw.rect(self.screen, (255, 0, 0), (self.state % self.grid_size * self.cell_size,
                                                     self.state // self.grid_size * self.cell_size,
                                                     self.cell_size, self.cell_size))

        # Draw passenger
        pygame.draw.circle(self.screen, (0, 255, 0), ((self.passenger_loc[1] * self.cell_size) + self.cell_size // 2,
                                                      (self.passenger_loc[0] * self.cell_size) + self.cell_size // 2),
                           self.cell_size // 4)

        # Draw destination
        pygame.draw.circle(self.screen, (0, 0, 255), ((self.destination_loc[1] * self.cell_size) + self.cell_size // 2,
                                                       (self.destination_loc[0] * self.cell_size) + self.cell_size // 2),
                           self.cell_size // 4)
        pygame.display.update()
        pygame.time.delay(delay) 

    def reset(self):
        """
        Reset the environment to the initial state and return it.
        Can be extended to reset passenger and destination locations.
        """
        self.state = 0  # Resets only the taxi's location for now
        self.in_transit = False 
        self.passenger_loc = (1, 2)
        self.destination_loc = (3, 4)
        self.obstacles = [3, 8,16,21]

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
            if self.state == self.passenger_loc[0] * self.grid_size + self.passenger_loc[1]:
                self.passenger_loc = (-1, -1)  # Passenger is now on-board
                self.in_transit = True  # Set flag to true when passenger is picked up
                reward = 10  # Reward for successful pick-up
            else:
                reward = -10  # Penalty for incorrect pick-up attempt
        elif action == 5:  # drop-off action
            if self.in_transit and self.state == self.destination_loc[0] * self.grid_size + self.destination_loc[1]:
                reward = 20  # Reward for successful drop-off and episode ends
                done = True
                self.in_transit = False  # Reset flag when passenger is dropped off
            elif self.in_transit:
                reward = -10  # Penalty for incorrect drop-off attempt
            else:
                reward = -20  # Penalty for attempting to drop off without a passenger

        else:
            raise ValueError("Invalid action")
        
        if self.state in self.obstacles:
            reward -= 20

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
            print("Passenger picked up!")
            self.passenger_loc = (-1, -1)  # Passenger is now on-board
            return 10, False  # Reward for successful pick-up
        else:
            print("Pickup attempted at wrong location.")
            return -10, False  # Penalty for incorrect pick-up
        
    

    def dropoff(self):
        """
        Handle the drop-off action and return the reward and done flag.
        """
        if self.state == self.destination_loc[0] * self.grid_size + self.destination_loc[1]:
            print("Passenger dropped off!")
            return 20, True  # Reward for successful drop-off and episode ends
        else:
            print("Dropoff attempted at wrong location.")
            return -20, False  # Penalty for incorrect drop-off

    def q_learning(self, num_episodes):
        clock = pygame.time.Clock()
        for episode in range(num_episodes):
            state = self.reset()  # Reset the environment at the start of each episode
            in_transit = False  # Flag to track if passenger has been picked up
            total_reward = 0
            done = False
            steps = 0  # Track the number of steps taken in the episode
            
            clock.tick(50)  # Limit FPS to 10

            while not done:
                self.draw_grid(delay=50) 
                pygame.display.update()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                # Determine which Q-table to use based on whether the passenger is in transit
                q_table_current = self.q_table_post_pickup if in_transit else self.q_table_pre_pickup
                action = np.random.choice(self.get_possible_actions()) if np.random.rand() < self.exploration_rate \
                    else np.argmax(q_table_current[state])

                next_state, reward, done = self.step(action)
                steps += 1  # Increment steps for each action taken
                
                # Update in_transit status based on the action taken
                if action == 4:  # Pickup action
                    in_transit = True
                elif action == 5 and in_transit:  # Successful drop-off
                    in_transit = False  # Reset for the next episode
                    

                # Update Q-value using the appropriate Q-table
                q_table_current[state, action] = (1 - self.learning_rate) * q_table_current[state, action] + \
                                                  self.learning_rate * (reward + self.discount_factor * np.max(q_table_current[next_state]))

                total_reward += reward
                state = next_state

            # Adjust the exploration rate
            self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay_rate)

            # Logging for every 100 episodes
            if episode % 100 == 0:
                print(f"Episode {episode}: Total Reward = {total_reward}, Steps = {steps}, Final Exploration Rate = {self.exploration_rate:.4f}")

        print("Training finished.")
        pygame.quit()
    
    
    def test_agent(self):
        state = self.reset()  # Start from the initial state for testing
        in_transit = False  # Initialize in_transit for testing phase
        total_reward = 0
        done = False
        steps = 0  # Track the number of steps during testing

        print("\nTesting trained agent...")
        while not done:
            # Choose the correct Q-table based on in_transit status
            q_table_current = self.q_table_post_pickup if in_transit else self.q_table_pre_pickup
            action = np.argmax(q_table_current[state])
            state, reward, done = self.step(action)
            total_reward += reward
            steps += 1
            print(f"Step {steps}: State = {state}, Action = {action}, Reward = {reward}")

            # Update in_transit status based on the action
            if action == 4:
                in_transit = True
            elif action == 5 and in_transit:
                in_transit = False

        print(f"Test Total Reward: {total_reward}, Total Steps: {steps}")
    
    
    


def plot_q_table(q_table, title="Q-Table"):
    # Assuming q_table is a numpy array of shape (num_states, num_actions)
    # We take the max of each row to get the best action value for each state
    max_q_values = np.max(q_table, axis=1)
    
    # Reshape the max Q-values back into the grid shape for visualization
    grid_shape = int(np.sqrt(len(max_q_values)))  # Assuming the grid is square
    heatmap_data = max_q_values.reshape((grid_shape, grid_shape))
    
    plt.figure(figsize=(8, 8))
    plt.title(title)
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.show()




# Main execution
if __name__ == "__main__":
    env = TaxiEnv()
    num_episodes = 1000
    env.q_learning(num_episodes)
    
    # After training, print or visualize the Q-tables:
    plot_q_table(env.q_table_pre_pickup, title="Pre-Pickup Q-Table")
    plot_q_table(env.q_table_post_pickup, title="Post-Pickup Q-Table")

    
    
    env.test_agent()  # Call the enhanced testing method
    

    # After training, test the trained agent
    state = env.reset()  # Start from the initial state for testing
    total_reward = 0
    done = False
    in_transit = False  # This is necessary to track the taxi's state regarding the passenger

    print("\nTesting trained agent...")
    while not done:
        # Choose the best action from the appropriate Q-table
        if in_transit:
            action = np.argmax(env.q_table_post_pickup[state])
        else:
            action = np.argmax(env.q_table_pre_pickup[state])
    
        state, reward, done = env.step(action)
    
    # Update in_transit status based on the action taken
        if action == 4:  # If the action was to pick up the passenger
            in_transit = True
        elif action == 5 and in_transit:  # If the action was to drop off the passenger
            in_transit = False  # Reset for possibly multiple tests or future extensions
    
        total_reward += reward

    print(f"Test Total Reward: {total_reward}")
