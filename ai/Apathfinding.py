import heapq
import math


class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent   #父节点
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f
    def __hash__(self):
        return hash(self.position)

    def __repr__(self):
        return f"Node(position: {self.position}, g: {self.g}, h: {self.h}, f: {self.f})"


def astar(maze, start, end):
    # 创建起始和结束节点
    start_node = Node(None, tuple(start))
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, tuple(end))
    end_node.g = end_node.h = end_node.f = 0
    open_list = []
    closed_list = set()

    # 将起始节点添加到open列表
    heapq.heappush(open_list, start_node)


    while open_list:
        current_node = heapq.heappop(open_list)
        closed_list.add(current_node)

        # 如果找到目标，构建路径
        if current_node == end_node:
            print('Find end !')
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1]  #


        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:  # 相邻方格
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])
            # 确保在迷宫范围内
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[0]) - 1) or node_position[1] < 0:
                continue
            # 确保可行走
            if maze[node_position[0]][node_position[1]] ==1:
                continue
            new_node = Node(current_node, node_position)
            children.append(new_node)


        for child in children:
            if child in closed_list:
                continue
            child.g = current_node.g + 1
            #欧几里得距离
            child.h = math.sqrt(
                (child.position[0] - end_node.position[0]) ** 2 + (child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h
            # 检查邻居节点是否已经在 OpenList 中
            found_in_open_list = False
            for open_node in open_list:
                if child == open_node:
                    found_in_open_list = True
                    # 检查是否有更好的路径
                    if child.g < open_node.g:
                        # 更新 OpenList 中节点的值
                        open_node.g = child.g
                        open_node.h = child.h
                        open_node.f = child.f
                        open_node.parent = current_node
                    break

            if not found_in_open_list:
                heapq.heappush(open_list, child)

    return None  # 如果没有找到路径