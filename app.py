import os
import sys
import time
import pygame
from states.menu import MainMenu

class App:
        def __init__(self):
            pygame.init()
            self.WIDTH,self.HEIGHT = 800, 600
            self.screen = pygame.display.set_mode((self.WIDTH,self.HEIGHT))
            self.running = True
            self.difficulty = 1
            self.state_stack = []
            self.load_states()

        def game_loop(self):
            while self.running:
                # self.get_dt()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                        pygame.quit()
                        sys.exit()
                self.update()
                self.render()

        def update(self):
            self.state_stack[-1].update()

        def render(self):
            self.state_stack[-1].render()
            # Render current state to the screen
            # self.screen.blit(pygame.transform.scale(self.game_canvas,(self.SCREEN_WIDTH, self.SCREEN_HEIGHT)), (0,0))
            pygame.display.flip()

        def load_states(self):
            self.main_menu = MainMenu(self)
            self.state_stack.append(self.main_menu)

if __name__ == "__main__":
    app = App()
    while app.running:
        app.game_loop()