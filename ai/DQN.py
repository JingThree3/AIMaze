import math
import random
from math import gamma

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple

from torch.cuda import device


# from main import maze


def action_to_dxdy(action):
    # 行为定义如下：
    # 0 - 上, 1 - 下, 2 - 左, 3 - 右
    if action == 0:  # 上
        return 0, -1
    elif action == 1:  # 下
        return 0, 1
    elif action == 2:  # 左
        return -1, 0
    elif action == 3:  # 右
        return 1, 0


class ReinforcementLearningEnvironment:
    def __init__(self, maze, player):
        self.maze = maze
        self.player = player
        self.visited_cells = set()  # 用于记录访问过的格子
        self.staypunish=0

    def get_state(self):
        # 返回当前状态，玩家的位置
        return self.player.x, self.player.y

    def get_reward(self, new_state):
        x, y = new_state  # 分解 new_state 为 x 和 y
        goal_x, goal_y = self.maze.get_end_pos()
        #print('x,y,gx,gy',x//self.maze.cell_size,y//self.maze.cell_size,goal_x,goal_y)
        distance = abs(goal_x - x//self.maze.cell_size) + abs(goal_y - y//self.maze.cell_size)
        #print('distance',goal_x - x//self.maze.cell_size,goal_y - y//self.maze.cell_size)
        # 根据距离计算奖励
        proximity_reward = max(0, 30- distance)
        print('Distance Reward',proximity_reward)
        if self.maze.is_goal(x, y):
            return 100 + proximity_reward+self.staypunish  # 到达终点
        elif self.maze.is_wall(x, y):
            return -50+self.staypunish    # 撞墙
        else:
            if (x, y) in self.visited_cells:
                return -0.1 + proximity_reward+self.staypunish
            else:
                self.visited_cells.add((x, y))  # 添加到访问记录中
                return 1 + proximity_reward + self.staypunish
                #print('Get A Walk')



        #print('No Behave')


    def is_done(self, new_state):
        # 判断游戏是否结束
        x,y=new_state
        return self.maze.is_goal(x,y)

    def step(self, action):
        dx, dy = action_to_dxdy(action)
        new_x = self.player.x + dx * self.maze.cell_size
        new_y = self.player.y + dy * self.maze.cell_size

        # 检查是否撞墙
        if self.player.collide_with_maze(new_x, new_y):
            # 如果撞墙，分配撞墙奖励并保持位置不变
            reward = -200+self.staypunish
            print('walalll')
            done = False
            new_state = (self.player.x, self.player.y)  # 保持在原地，未移动
        else:
            # 如果没有撞墙，正常移动玩家
            self.player.move(dx, dy)
            new_state = (self.player.x, self.player.y)  # 更新为新位置
            reward = self.get_reward(new_state)
            print('reward',reward)
            done = self.is_done(new_state)

        return new_state, reward, done
    def reset(self):
        # 重置玩家到起始位置
        self.player.reset_player()
        # 返回初始状态
        return self.get_state()

    def is_success(self):
        return self.maze.is_goal(self.player.x, self.player.y)

#----------------DQN部分-------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化 DQN 模型的参数
input_size = 2   # 玩家位置
hidden_size = 64  # 隐藏层的大小
output_size = 4  # 上下左右
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.gamma = 0.9
        # 定义第一个全连接层 接受输入玩家状态
        self.fc1 = nn.Linear(input_size, hidden_size)
        # 定义激活函数ReLU
        self.relu = nn.ReLU()   #f(x) = max(0, x)
        # 定义第二个全连接层 输出四个动作的预期 Q 值
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 前向传播过程：将输入数据通过网络层和激活函数
        x = self.relu(self.fc1(x))  # 通过第一个全连接层然后应用 ReLU 激活函数
        x = self.fc2(x)  # 最后通过第二个全连接层
        return x

    def predict_action(self, state, epsilon):
        state_tensor = torch.tensor([state], dtype=torch.float).to(device)  # 转换为张量
        if random.random() > epsilon:
            with torch.no_grad():
                q_values = self(state_tensor)
                return q_values.max(1)[1].item()  # 返回具有最大 Q 值的动作
        else:
            return random.randrange(output_size)  # 随机选择一个动作

    def learn(self, optimizer, criterion, experiences, target_model):
        # 分解经验
        states, actions, rewards, next_states = zip(*experiences)

        # 转换为张量
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)

        # 获取当前状态的预测 Q 值
        current_q_values = self(states).gather(1, actions.unsqueeze(-1))

        # 使用目标网络计算下一个状态的最大 Q 值
        max_next_q_values = target_model(next_states).detach().max(1)[0]
        expected_q_values = rewards + (self.gamma * max_next_q_values)

        # 计算损失
        loss = criterion(current_q_values, expected_q_values.unsqueeze(1))

        # 优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        #存储经验
        self.memory.append(Experience(*args))


    def sample(self, batch_size):
        # 随机抽取一批经验用于训练
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
