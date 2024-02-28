import random

import pygame


class Maze:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.cell_size = 40  # 每个迷宫单元的大小
        self.maze_size = 10
        self.start_pos = (1, 1)
        self.end_pos = (self.maze_size - 2, self.maze_size - 2)
        # 初始化迷宫布局
        self.maze_layout = [[0 for _ in range(self.maze_size)] for _ in range(self.maze_size)]
        self.generate_maze()
        self.flag = [[0 for _ in range(self.maze_size)] for _ in range(self.maze_size)]

    def generate_maze(self):
        # 生成迷宫的边界墙壁
        for i in range(self.maze_size):
            for j in range(self.maze_size):
                if i == 0 or j == 0 or i == self.maze_size - 1 or j == self.maze_size - 1:
                    self.maze_layout[i][j] = 1
                else:
                    # 除边界外其它位置先设置为可通行
                    self.maze_layout[i][j] = 0
        '''
        self.maze_layout=[[1, 1, 1, 1, 1, 1],
                          [1, 0, 1, 0, 0, 1],
                          [1, 0, 0, 0, 1, 1],
                          [1, 1, 0, 0, 0, 1],
                          [1, 0, 0, 0, -1, 1],
                            [1, 1, 1, 1, 1, 1]]
        '''
        # 随机生成内部墙壁
        for _ in range(int((self.maze_size - 2) ** 2 * 0.2)):  # 20%的内部墙壁
            while True:
                x, y = random.randint(1, self.maze_size - 2), random.randint(1, self.maze_size - 2)
                # 确保不在起始点和终点生成墙壁
                if (x, y) not in [(1, 1), (self.maze_size - 2, self.maze_size - 2)]:
                    if self.maze_layout[x][y] == 0:  # 确保选中的位置不是墙壁
                        self.maze_layout[x][y] = 1
                        break

        # 设置终点
        self.maze_layout[self.maze_size - 2][self.maze_size - 2] = -1

    def reset_maze(self):
        self.maze_layout = [[0 for _ in range(self.maze_size)] for _ in range(self.maze_size)]
        self.flag = [[0 for _ in range(self.maze_size)] for _ in range(self.maze_size)]
        self.generate_maze()

    def is_wall(self, x, y):
        # 将像素位置转换为单元格索引
        cell_x = x // self.cell_size
        cell_y = y // self.cell_size
        if 0 <= cell_x < self.maze_size and 0 <= cell_y < self.maze_size:
            return self.maze_layout[cell_y][cell_x] == 1
        return False

    def is_goal(self, x, y):

        cell_x = x // self.cell_size
        cell_y = y // self.cell_size
        if 0 <= cell_x < self.maze_size and 0 <= cell_y < self.maze_size:
            return self.maze_layout[cell_y][cell_x]  == -1
        return False

    def get_start_pos(self):
        return self.start_pos

    def get_end_pos(self):
        return self.end_pos

    def draw(self, screen):
        for i, row in enumerate(self.maze_layout):
            for j, cell in enumerate(row):
                x, y = j * self.cell_size, i * self.cell_size
                if cell == 1:
                    # 绘制带X的墙壁
                    pygame.draw.rect(screen, (255, 255, 255), (x, y, self.cell_size, self.cell_size), 1)
                    pygame.draw.line(screen, (255, 255, 255), (x, y), (x + self.cell_size, y + self.cell_size))
                    pygame.draw.line(screen, (255, 255, 255), (x + self.cell_size, y), (x, y + self.cell_size))
                elif cell == 0:
                    # 绘制可通行区域
                    pygame.draw.rect(screen, (255, 255, 255), (x, y, self.cell_size, self.cell_size))
                elif cell == -1:
                    # 绘制终点
                    pygame.draw.circle(screen, (0, 255, 0), (x + self.cell_size // 2, y + self.cell_size // 2),
                                       self.cell_size // 2 - 2)

                    # 标记a*算法路径
                font = pygame.font.Font(None, 36)
                if self.flag[i][j] == 2:
                    text_surface = font.render(str(cell), True, (255, 0, 255))
                    text_rect = text_surface.get_rect(center=(x + self.cell_size // 2, y + self.cell_size // 2))
                    screen.blit(text_surface, text_rect)
