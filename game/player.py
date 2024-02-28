import math

import pygame

class Player:
    def __init__(self, maze):
        self.cell_size = maze.cell_size
        self.x = self.cell_size  # 初始化玩家位置0,0
        self.y = self.cell_size
        self.color = (255, 0, 0)
        self.maze = maze
        self.moving = False

    def update(self, events):
        if not self.moving:
            for event in events:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.move(-1, 0)
                    elif event.key == pygame.K_RIGHT:
                        self.move(1, 0)
                    elif event.key == pygame.K_UP:
                        self.move(0, -1)
                    elif event.key == pygame.K_DOWN:
                        self.move(0, 1)

    def move(self, dx, dy):
        new_x = self.x + dx * self.cell_size
        new_y = self.y + dy * self.cell_size

        if not self.collide_with_maze(new_x, new_y):
            self.x = new_x
            self.y = new_y
            self.moving = True

            # 检查是否到达终点
            if self.at_maze_exit():
                print("已到达终点！")

    def reset_player(self):
        self.x = self.cell_size
        self.y = self.cell_size
    def has_won(self):
        return self.at_maze_exit()
    def collide_with_maze(self, x, y):
        maze_x = x // self.cell_size
        maze_y = y // self.cell_size

        if 0 <= maze_x < self.maze.maze_size and 0 <= maze_y < self.maze.maze_size:
            if self.maze.maze_layout[maze_y][maze_x] == 1:
                return True
        return False

    def at_maze_exit(self):
        maze_x = self.x // self.cell_size
        maze_y = self.y // self.cell_size

        if self.maze.maze_layout[maze_y][maze_x] == -1:
            return True
        return False

    def draw(self, screen):
        center = (self.x + self.cell_size // 2, self.y + self.cell_size // 2)
        points = []
        for i in range(5):
            angle = math.radians(72 * i - 90)
            x = center[0] + self.cell_size // 3 * math.cos(angle)
            y = center[1] + self.cell_size // 3 * math.sin(angle)
            points.append((x, y))
        pygame.draw.polygon(screen, self.color, points)