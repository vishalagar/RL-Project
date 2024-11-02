import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import sys
import os

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 0, 0)
GREEN = (0, 200, 0)

# Game settings
BLOCK_SIZE = 20
SPEED = 100  # Default speed for AI

# Initialize Pygame
pygame.init()
font = pygame.font.SysFont('timenewroman', 25)

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
DIRECTIONS = [LEFT, RIGHT, UP, DOWN]

class Snake:
    def __init__(self):
        self.body = [((SCREEN_WIDTH // 2), (SCREEN_HEIGHT // 2))]
        self.direction = random.choice(DIRECTIONS)
        self.grow = False

    def move(self):
        x, y = self.direction
        head_x, head_y = self.body[0]
        new_head = (head_x + x * BLOCK_SIZE, head_y + y * BLOCK_SIZE)
        self.body.insert(0, new_head)
        if not self.grow:
            self.body.pop()
        else:
            self.grow = False

    def change_direction(self, new_direction):
        opposite_direction = (-self.direction[0], -self.direction[1])
        if new_direction != opposite_direction:
            self.direction = new_direction

    def check_collision(self):
        head = self.body[0]
        # Check wall collision
        if (head[0] < 0 or head[0] >= SCREEN_WIDTH or
            head[1] < 0 or head[1] >= SCREEN_HEIGHT):
            return True
        # Check self collision
        if head in self.body[1:]:
            return True
        return False

class Food:
    def __init__(self):
        self.position = (0, 0)
        self.randomize_position()

    def randomize_position(self, snake_body=None):
        while True:
            x = random.randint(0, (SCREEN_WIDTH - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (SCREEN_HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            if snake_body and (x, y) in snake_body:
                continue
            else:
                self.position = (x, y)
                break

class DQN(nn.Module):
    def __init__(self, input_size=11, hidden_size=256, output_size=3):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class Agent:
    def __init__(self, model=None):
        self.n_games = 0
        self.epsilon = .1  # Randomness
        self.gamma = 0.85  # Discount rate
        self.memory = deque(maxlen=100_000)
        self.model = model if model else DQN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.record = 0  # Keep track of the best score

    def get_state(self, game):
        snake = game.snake
        head = snake.body[0]
        point_l = (head[0] - BLOCK_SIZE, head[1])
        point_r = (head[0] + BLOCK_SIZE, head[1])
        point_u = (head[0], head[1] - BLOCK_SIZE)
        point_d = (head[0], head[1] + BLOCK_SIZE)

        dir_l = snake.direction == LEFT
        dir_r = snake.direction == RIGHT
        dir_u = snake.direction == UP
        dir_d = snake.direction == DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.position[0] < head[0],  # Food left
            game.food.position[0] > head[0],  # Food right
            game.food.position[1] < head[1],  # Food up
            game.food.position[1] > head[1],  # Food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > 1_000:
            mini_sample = random.sample(self.memory, 1_000)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)

        self.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.train_step([state], [action], [reward], [next_state], [done])

    def train_step(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(np.array(states), dtype=torch.float)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)

        # Predict Q values
        pred = self.model(states)
        target = pred.clone().detach()

        for idx in range(len(dones)):
            Q_new = rewards[idx]
            if not dones[idx]:
                Q_new = rewards[idx] + self.gamma * torch.max(self.model(next_states[idx]))
            target[idx][actions[idx]] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()

    def get_action(self, state):
        # Random moves: tradeoff between exploration and exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

class Game:
    def __init__(self, human=False, high_score=0):
        self.display = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        if not human:
            pygame.display.set_caption('Snake AI')
            self.speed = SPEED  # Default speed for AI
        else:
            pygame.display.set_caption('Snake Human')
            self.speed = 10  # Slower speed for human player
        self.clock = pygame.time.Clock()
        self.reset()
        self.human = human
        self.high_score = high_score  # Initialize high score

    def reset(self):
        self.snake = Snake()
        self.food = Food()
        self.food.randomize_position(self.snake.body)
        self.score = 0
        self.frame_iteration = 0  # To prevent infinite loops

    def is_collision(self, point=None):
        if point is None:
            point = self.snake.body[0]
        # Check wall collision
        if (point[0] < 0 or point[0] >= SCREEN_WIDTH or
            point[1] < 0 or point[1] >= SCREEN_HEIGHT):
            return True
        # Check self collision
        if point in self.snake.body[1:]:
            return True
        return False

    def play_step(self, action=None):
        self.frame_iteration += 1
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if self.human:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.snake.change_direction(LEFT)
                    elif event.key == pygame.K_RIGHT:
                        self.snake.change_direction(RIGHT)
                    elif event.key == pygame.K_UP:
                        self.snake.change_direction(UP)
                    elif event.key == pygame.K_DOWN:
                        self.snake.change_direction(DOWN)

        # Move the snake
        if not self.human:
            self._move(action)
        self.snake.move()

        # Check if game over
        game_over = False
        reward = 0
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake.body):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # Check if food eaten
        if self.snake.body[0] == self.food.position:
            self.snake.grow = True
            self.score += 1
            reward = 10
            self.food.randomize_position(self.snake.body)
            # Update high score if needed
            if self.score > self.high_score:
                self.high_score = self.score

        # Update UI and clock
        self._update_ui()
        self.clock.tick(self.speed)  # Use self.speed instead of SPEED

        return reward, game_over, self.score

    def _move(self, action):
        # [straight, right, left]
        clock_wise = [RIGHT, DOWN, LEFT, UP]
        idx = clock_wise.index(self.snake.direction)

        if np.array_equal(action, [1, 0, 0]):
            # Keep the same direction
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            # Turn right
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:  # [0, 0, 1]
            # Turn left
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        self.snake.change_direction(new_dir)

    def _update_ui(self):
        self.display.fill(BLACK)

        for pos in self.snake.body:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(pos[0], pos[1], BLOCK_SIZE, BLOCK_SIZE))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.position[0], self.food.position[1], BLOCK_SIZE, BLOCK_SIZE))

        # Display current score
        text_score = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text_score, [0, 0])

        # Display high score
        text_high_score = font.render("High Score: " + str(self.high_score), True, WHITE)
        self.display.blit(text_high_score, [0, 30])  # Position it below the current score

        pygame.display.flip()

def train_ai(agent):
    total_score = 0
    game = Game(high_score=agent.record)
    while True:
        # Get old state
        state_old = agent.get_state(game)

        # Get move
        final_move = agent.get_action(state_old)

        # Perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Train short memory
        agent.train_short_memory(state_old, np.argmax(final_move), reward, state_new, done)

        # Remember
        agent.remember(state_old, np.argmax(final_move), reward, state_new, done)

        if done:
            # Train long memory
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > agent.record:
                agent.record = score
                game.high_score = score
                # Save the model
                torch.save(agent.model.state_dict(), 'best_model.pth')
                print(f'New best score: {score}. Model saved as best_model.pth')

            print('Game', agent.n_games, 'Score', score, 'Record:', agent.record)

def play_ai(model_name='best_model.pth'):
    agent = Agent()
    if os.path.isfile(model_name):
        agent.model.load_state_dict(torch.load(model_name))  # Removed weights_only=True
        print(f'Loaded model: {model_name}')
    else:
        print(f'Model file {model_name} not found.')
        return

    game = Game(high_score=agent.record)
    while True:
        # Get state
        state = agent.get_state(game)

        # Get action
        final_move = agent.get_action(state)

        # Perform move
        reward, done, score = game.play_step(final_move)

        if done:
            # Update high score
            if score > game.high_score:
                game.high_score = score
            game.reset()
            agent.n_games += 1
            print('Game', agent.n_games, 'Score:', score, 'High Score:', game.high_score)

def play_human():
    game = Game(human=True)
    while True:
        reward, done, score = game.play_step()
        if done:
            # Update high score
            if score > game.high_score:
                game.high_score = score
            game.reset()
            print('Score:', score, 'High Score:', game.high_score)

def main():
    print("Welcome to the Snake Game!")
    print("Enter '1' to play manually (Human), '2' for AI mode:")
    mode = input("Your choice: ")
    if mode == '1':
        play_human()
    elif mode == '2':
        print("AI Mode Selected.")
        print("Choose an option:")
        print("1. Start fresh")
        print("2. Load best score model")
        print("3. Load specific model")
        ai_choice = input("Your choice: ")
        if ai_choice == '1':
            print("Starting fresh training...")
            agent = Agent()
            train_ai(agent)
        elif ai_choice == '2':
            print("Loading best score model...")
            agent = Agent()
            if os.path.isfile('best_model.pth'):
                agent.model.load_state_dict(torch.load('best_model.pth'))  # Removed weights_only=True
                agent.record = 0  # Reset record if needed
                train_ai(agent)
            else:
                print("No best model found. Starting fresh training...")
                train_ai(agent)
        elif ai_choice == '3':
            model_name = input("Enter the model filename (e.g., model.pth): ")
            if os.path.isfile(model_name):
                print(f"Loading model {model_name}...")
                agent = Agent()
                agent.model.load_state_dict(torch.load(model_name))  # Removed weights_only=True
                train_ai(agent)
            else:
                print(f"Model file {model_name} not found. Starting fresh training...")
                agent = Agent()
                train_ai(agent)
        else:
            print("Invalid input. Exiting...")
    else:
        print("Invalid input. Exiting...")

if __name__ == '__main__':
    main()
