from collections import defaultdict

import pygame
import torch.optim as optim
import torch.nn as nn
import random
import torch
from ai.Apathfinding import astar
from ai.DQN import ReinforcementLearningEnvironment, DQN, input_size, hidden_size, output_size, ReplayMemory


class PlayerController:
    def __init__(self, player):
        self.player = player
        self.completed = False

    def update(self, events):
        self.player.update(events)
        if self.player.has_won():
            self.completed= True

class AStarController:
    def __init__(self, maze, player, start_pos, end_pos):
        self.completed = False
        self.path_found = True
        self.maze = maze
        self.player = player
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.path = None
        self.path_nodes = []
        self.last_move_time = pygame.time.get_ticks()  # 记录上次移动的时间
        self.move_delay = 200  # 每次移动之间的延迟时间
        self.current_position=[1,1]
        self.current_node_index = 0  # 当前节点索引

    def update(self):
        # 如果路径为空，调用 A* 算法来获得路径
        if self.path is None:
            self.calculate_path()
        if self.player.has_won():
            self.completed = True

    def visiualPath(self):
        # 检查是否已经到达更新时间
        current_time = pygame.time.get_ticks()
        # 检查是否到达更新时间
        if current_time - self.last_move_time >= self.move_delay:
            if self.current_node_index < len(self.path_nodes):
                next_position = self.path_nodes[self.current_node_index]
                print(next_position)
                dx = next_position[0] - self.current_position[0]
                dy = next_position[1] - self.current_position[1]
                self.player.move(dy, dx)  # 移动玩家
                self.last_move_time = current_time  # 更新上次移动时间
                self.current_position = next_position  # 更新当前位置
                self.current_node_index += 1  # 移动到下一个节点

    def reset_controller(self):
        self.path = None
        self.path_nodes = []
        self.current_node_index = 0
        self.last_move_time = pygame.time.get_ticks()
        self.current_position = [1, 1]

    def calculate_path(self):
        self.path = astar(self.maze.maze_layout, self.start_pos, self.end_pos)
        self.path_nodes = []
        if self.path:
            self.path_found = True
            for position in self.path:
                # 将路径上的节点添加到path_nodes列表
                self.path_nodes.append(position)
                self.maze.flag[position[0]][position[1]] = 2
            self.path_nodes.pop(0)  # 移除起始节点

        else:
            self.path_nodes = []
            self.path_found = False
       # self.path_nodes = self.path[1:] if self.path else []  # 跳过起始节点

class QLearningController:
    def __init__(self, maze, player):
        self.completed = False
        self.maze = maze
        self.player = player
        self.rl_environment = ReinforcementLearningEnvironment(maze, player)

        self.endState = 0;  # -1 fail   1 suc
        self.stay_count = 0  # 记录在同一格子停留的次数
        self.stay_threshold = 30  # 设定停留阈值


        self.q_table = defaultdict(lambda: defaultdict(lambda: 0))

        self.alpha = 0.1  # 学习率
        self.gamma = 0.9  # 折扣因子

        self.success_count = 0  # 成功次数
        self.success_threshold = 2  # 成功阈值
        self.fail_count = 0  # 游戏失败次数
        self.fail_threshold = 5  # 失败阈值

        self.epsilon = 1.0  # 初始 epsilon 值
        self.epsilon_decay = 0.99  # epsilon 衰减率
        self.epsilon_min = 0.01  # epsilon 的最小值

        self.last_position = None

    def choose_action(self, state):
        q_values = self.q_table[state]
        if random.random() < self.epsilon:
            return random.choice([0, 1, 2, 3])  # 随机选择
        else:
            # 如果 q_values 为空，则随机选择动作，否则选择最大 Q 值的动作
            if not q_values:
                return random.choice([0, 1, 2, 3])
            max_q = max(q_values.values())
            actions_with_max_q = [action for action, value in q_values.items() if value == max_q]
            return random.choice(actions_with_max_q)
    def update_q_table(self, state, action, reward, next_state, done):
        old_value = self.q_table[state][action]
        # 检查 next_state 是否存在于 Q 表中，如果不存在，则自动创建
        next_max = max(self.q_table[next_state].values(), default=0) if not done else 0
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[state][action] = new_value

    def update(self):
        current_state = self.rl_environment.get_state()
        if self.last_position == current_state:
            self.stay_count += 1
            self.rl_environment.staypunish = -10  # 静止惩罚
            if self.stay_count >= self.stay_threshold:
                print("Game failed: Player stayed too long in the same cell.")
                self.fail_count += 1
                self.rl_environment.reset()
                if self.fail_count >= self.fail_threshold:
                    print("Training ended due to too many failures.")
                    self.completed = True
                    self.endState = -1
                    return True
                self.stay_count = 0
                self.rl_environment.staypunish = 0
                self.last_position = None
                return False
        else:
            self.stay_count = 0
            self.rl_environment.staypunish =0

        action = self.choose_action(current_state)
        new_state, reward, done = self.rl_environment.step(action)
        self.update_q_table(current_state, action, reward, new_state, done)
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)
        self.last_position = current_state

        if done:
            if self.rl_environment.is_success():
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.completed = True
                    print('QLearning Success!')
                    self.endState = 1
                    return True
            else:
                self.success_count = 0

            self.rl_environment.reset()

        return False

    def reset_controller(self):
        self.rl_environment.reset()

        # 重置玩家位置和状态
        self.completed = False
        self.epsilon = 1.0
        self.last_position = None
        # 清零计数器
        self.stay_count = 0
        self.fail_count = 0
        self.success_count = 0


class DeepQNetworkController:
    def __init__(self, maze, player):
        self.completed = False
        self.maze = maze
        self.player = player
        self.rl_environment = ReinforcementLearningEnvironment(maze, player)

        self.endState=0;  #-1 fail   1 suc
        self.stay_count = 0  # 记录在同一格子停留的次数
        self.stay_threshold = 30  # 设定停留阈值
        self.replay_memory = ReplayMemory(10000)
        # 两个网络
        self.model = DQN(input_size, hidden_size, output_size)
        self.target_model = DQN(input_size, hidden_size, output_size)

        # 使用 Adam 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        # 使用均方误差作为损失函数
        self.criterion = nn.MSELoss()

        self.success_count = 0  # 连续成功次数计数器
        self.success_threshold = 2  # 连续成功的阈值
        self.fail_count = 0  # 记录游戏失败次数
        self.fail_threshold = 5  # 设定失败的阈值
        self.epsilon = 1.0  # 初始 epsilon 值
        self.epsilon_decay = 0.99  # epsilon 衰减率
        self.epsilon_min = 0.01  # epsilon 的最小值

        self.update_count = 0 #迭代计数
        self.last_position = None


    def update_target_model(self):
        # 同步主网络和目标网络的参数
        self.target_model.load_state_dict(self.model.state_dict())
    def update(self):
        #更新目标网络参数
        self.update_count += 1
        if self.update_count % 10 == 0:
            self.update_target_model()
            self.update_count = 0  # 重置计数器

        # 获取当前状态
        current_state = self.rl_environment.get_state()
        #print('--------------------')
        #print('current',current_state)
        #print('last',self.last_position)
        if self.last_position == current_state:
            self.stay_count += 1
            self.rl_environment.staypunish=-10  #静止惩罚
            if self.stay_count >= self.stay_threshold:
                print("Game failed: Player stayed too long in the same cell.")
                self.fail_count += 1  # 增加失败计数
                print('fail-----',self.fail_count)
                self.rl_environment.reset()  # 重置环境
                if self.fail_count >= self.fail_threshold:
                    print('fail',self.fail_count,self.fail_threshold)
                    print("Training ended due to too many failures.")
                    self.completed=True
                    self.endState=-1
                    return True  # 超过失败阈值，训练结束

                #self.rl_environment.reset()  # 重置环境
                self.stay_count = 0  # 重置停留计数器
                self.rl_environment.staypunish =0
                self.last_position = None  # 重置最后位置
                return False  # 表示游戏失败 继续下一轮训练
        else:
            self.rl_environment.staypunish =0
            self.stay_count = 0  # 重置停留计数器


        # 从模型中获取动作
        action = self.model.predict_action(current_state, self.epsilon)
        # 执行动作并获取反馈
        new_state, reward, done = self.rl_environment.step(action)
        # 将经验存储到经验回放池中
        self.replay_memory.push(current_state, action, reward, new_state)

        BATCH_SIZE=5
        #经验池的经验积累到一定程度开始训练主网络
        if len(self.replay_memory) > BATCH_SIZE:
            experiences = self.replay_memory.sample(BATCH_SIZE)
            self.model.learn(self.optimizer, self.criterion, experiences, self.target_model)

        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

        self.last_position = current_state

        # 检查是否完成
        if done:
            if self.rl_environment.is_success():
                self.success_count += 1

                if self.success_count >= self.success_threshold:
                    self.completed = True
                    print('DQN Success!')
                    self.endState=1
                    return True  # 达到连续成功的阈值，训练结束
            else:
                self.success_count = 0  # 重置连续成功计数器

            self.rl_environment.reset()

        return False  # 继续训练
    def reset_controller(self):
        self.rl_environment.reset()

        # 重置玩家位置和状态
        self.completed = False
        self.epsilon = 1.0
        self.last_position = None
        # 清零计数器
        self.stay_count = 0
        self.fail_count = 0
        self.success_count = 0

