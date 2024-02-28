import pygame
from ai.Apathfinding import astar
from game.ControllerList import *
from game.maze import Maze
from game.player import Player
def create_button(screen, text, position, size):
    font = pygame.font.SysFont(None, 36)
    button = pygame.Rect(position, size)
    pygame.draw.rect(screen, (255, 0, 0), button)  # 设置按钮颜色
    text_surf = font.render(text, True, (255, 255, 255))
    text_rect = text_surf.get_rect(center=button.center)
    screen.blit(text_surf, text_rect)
    return button
def restart_game(maze, player, controller, selected_mode):
    maze.reset_maze()  # 重置迷宫
    player.reset_player()  # 重置玩家位置
    game_won = False
    game_lost = False
    # A*
    if selected_mode == 'A* Algorithm':
        controller.reset_controller()
        controller.calculate_path()
        controller.completed = False

    elif selected_mode == 'Deep Q Network'  or selected_mode == 'Q Learning':
        print('reset')
        controller.reset_controller()


def show_menu(screen):
    menu_running = True
    modes = ['Player Control', 'A* Algorithm', 'Q Learning', 'Deep Q Network']
    buttons = []
    button_positions = [(100, 50 + i * 60) for i in range(len(modes))]
    button_size = (300, 50)

    # 创建按钮的矩形对象
    for position in button_positions:
        buttons.append(pygame.Rect(position, button_size))

    while menu_running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.MOUSEBUTTONDOWN:
                for i, button in enumerate(buttons):
                    if button.collidepoint(event.pos):
                        return modes[i]

        # 绘制菜单背景
        screen.fill((0, 0, 0))

        # 绘制按钮和文本
        for i, button in enumerate(buttons):
            create_button(screen, modes[i], button.topleft, button.size)

        pygame.display.flip()
def show_message(screen, text, position, size,fontsize=46):
    font = pygame.font.SysFont(None, fontsize)
    text_surf = font.render(text, True, (255, 215, 0))  # 选择一个颜色，例如金色
    text_rect = text_surf.get_rect(center=position)
    pygame.draw.rect(screen, (0, 0, 0), text_rect.inflate(20, 10))  # 绘制文本背景
    screen.blit(text_surf, text_rect)


#------------游戏主要流程---------------#


# 初始化
pygame.init()
screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Maze Game')

selected_mode = show_menu(screen)
if selected_mode is None:
    pygame.quit()
    exit()


# 创建迷宫和玩家
maze = Maze(screen_width, screen_height)
player = Player(maze)
start_pos = maze.get_start_pos()
end_pos = maze.get_end_pos()

# 根据所选模式设置控制器
if selected_mode == 'Player Control':
    controller = PlayerController(player)
elif selected_mode == 'A* Algorithm':
    controller = AStarController(maze, player, start_pos, end_pos)
elif selected_mode == 'Q Learning':
     controller = QLearningController(maze, player)
elif selected_mode == 'Deep Q Network':
    controller = DeepQNetworkController(maze, player)



# 游戏主循环
running = True
game_won = False
game_lost = False
restart_button = create_button(screen, 'Restart', (700, 550), (100, 40))  # restart按钮

while running:
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            #if selected_mode !='Reinforcement Learning':
                if restart_button.collidepoint(event.pos) :
                    if selected_mode =='Player Control' or selected_mode =='A* Algorithm':
                        if controller.completed:
                            restart_game(maze, player, controller, selected_mode)
                            game_won = False  # 重置游戏通关状态
                            game_lost = False  # 重置失败状态
                    else:
                        restart_game(maze, player, controller, selected_mode)
                        game_won = False  # 重置游戏通关状态
                        game_lost = False  # 重置失败状态


    # 玩家控制时重置移动状态
    if selected_mode == 'Player Control':
        player.moving = False
        controller.update(events)


    if selected_mode == 'Player Control':
        controller.update(events)
    elif selected_mode == 'A* Algorithm':
        controller.update()
        controller.visiualPath()
        pass
    elif selected_mode == 'Q Learning':
        if not controller.completed:
            if not game_lost and not game_won:
                if controller.update():  # 如果返回 True，则训练结束
                    if controller.completed:
                        if controller.endState == 1:
                            print("Training finished successfully.")
                            game_won = True
                        elif controller.endState == -1:
                            print("Training finished with too many failures.")
                            game_lost = True
    elif selected_mode == 'Deep Q Network':
        if not controller.completed:
            if not game_lost and not game_won:
                if controller.update():  # 如果返回 True，则训练结束
                    if controller.completed:
                        if controller.endState == 1:
                            print("Training finished successfully.")
                            game_won = True
                        elif controller.endState ==-1:
                            print("Training finished with too many failures.")
                            game_lost = True

    # 更新
    #player.update(events)
    if player.has_won():
        game_won = True  # 玩家到达终点


    # 绘制
    screen.fill((0, 0, 0))
    maze.draw(screen)
    player.draw(screen)

    if selected_mode == 'Deep Q Network' or selected_mode =='Q Learning':
        show_message(screen,'Success Count',(screen.get_width()-150 , 20), (100, 20),25)
        show_message(screen, 'Failure Count', (screen.get_width() - 150, 50), (100, 20), 25)
        show_message(screen, controller.success_count.__str__(), (screen.get_width() - 50, 20), (100, 20), 25)
        show_message(screen, controller.fail_count.__str__(), (screen.get_width() - 50, 50), (100, 20), 25)

    restart_button = create_button(screen, 'Restart', (700, 550), (100, 40))

    if selected_mode == 'A* Algorithm' and not controller.path_found:
        # 显示未找到路径的消息
        show_message(screen, 'No Path', (screen.get_width() // 2, screen.get_height() // 2), (200, 50))

        continue  # 跳过其他渲染逻辑

    if game_won:
        show_message(screen, 'You Win!', (screen.get_width() // 2, screen.get_height() // 2), (200, 50))
    elif game_lost:
        show_message(screen, 'You Lose!', (screen.get_width() // 2, screen.get_height() // 2), (200, 50))

    # 刷新屏幕
    pygame.display.flip()



pygame.quit()




