import pygame
import numpy as np

class TaxiEnv:
    def __init__(self):
        # Edit these values to customize the program
        self.grid_size = 5
        self.cell_size = 100  # size of each cell in pixels
        self.num_states = self.grid_size ** 2
        self.num_actions = 6  # 4 directions + pick-up + drop-off
        self.state = 0  # initial state
        self.passenger_loc = (1, 2)  # passenger initial location
        self.destination_loc = (3, 4)  # destination initial location
        self.q_table = np.zeros((self.num_states, self.num_actions))
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 1.0
        self.min_exploration_rate = 0.01
        self.exploration_decay_rate = 0.99

        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.grid_size * self.cell_size, self.grid_size * self.cell_size))
        pygame.display.set_caption('Taxi Environment')

    def draw_grid(self):
        self.screen.fill((255, 255, 255))  # white background

        # Draw grid lines
        for x in range(0, self.grid_size * self.cell_size, self.cell_size):
            pygame.draw.line(self.screen, (0, 0, 0), (x, 0), (x, self.grid_size * self.cell_size))
        for y in range(0, self.grid_size * self.cell_size, self.cell_size):
            pygame.draw.line(self.screen, (0, 0, 0), (0, y), (self.grid_size * self.cell_size, y))

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

    def get_possible_actions(self):
        return [0, 1, 2, 3, 4, 5]  # 0: up, 1: down, 2: left, 3: right, 4: pick-up, 5: drop-off

    def step(self, action):
        if action == 0:  # move up
            new_state = self.state - self.grid_size if self.state >= self.grid_size else self.state
        elif action == 1:  # move down
            new_state = self.state + self.grid_size if self.state < self.num_states - self.grid_size else self.state
        elif action == 2:  # move left
            new_state = self.state - 1 if self.state % self.grid_size != 0 else self.state
        elif action == 3:  # move right
            new_state = self.state + 1 if (self.state + 1) % self.grid_size != 0 else self.state
        elif action == 4:  # pick-up passenger
            if self.state == self.passenger_loc[0] * self.grid_size + self.passenger_loc[1]:
                self.passenger_loc = (-1, -1)  # passenger picked up
                reward = 10
            else:
                reward = -10
            return self.state, reward
        elif action == 5:  # drop-off passenger
            if self.state == self.destination_loc[0] * self.grid_size + self.destination_loc[1]:
                reward = 20
            else:
                reward = -20
            return self.state, reward
        else:
            raise ValueError("Invalid action")

        self.state = new_state
        reward = -1  # movement cost
        return self.state, reward

    def q_learning(self, num_episodes):
        clock = pygame.time.Clock()

        for episode in range(num_episodes):
            state = self.state
            total_reward = 0

            for _ in range(100):  # Limiting steps per episode to avoid infinite loop
                self.draw_grid()
                pygame.display.update()

                clock.tick(10)  # Limit FPS to 10

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

                possible_actions = self.get_possible_actions()
                if np.random.rand() < self.exploration_rate:
                    action = np.random.choice(possible_actions)
                else:
                    action = np.argmax(self.q_table[state])

                next_state, reward = self.step(action)

                old_q_value = self.q_table[state, action]
                next_max_q_value = np.max(self.q_table[next_state])

                new_q_value = (1 - self.learning_rate) * old_q_value + \
                              self.learning_rate * (reward + self.discount_factor * next_max_q_value)

                self.q_table[state, action] = new_q_value

                total_reward += reward
                state = next_state

                if reward == 20 or reward == -10:  # episode ends
                    break

            self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay_rate)

            if episode % 100 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward}")

        print("Training finished.")
        pygame.quit()

if __name__ == "__main__":
    env = TaxiEnv()
    num_episodes = 1000
    env.q_learning(num_episodes)
