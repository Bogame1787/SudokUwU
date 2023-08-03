import pygame
import sys
from widgets.colors import *

class Button:
    def __init__(self, x, y, width, height, text, font_size=30, color=BLACK, border_color = BLACK, highlight_color = GREEN_HIGHLIGHT, text_color = WHITE, callback=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.highlight_color = highlight_color
        self.text = text
        self.font_size = font_size
        self.is_hovered = False
        self.transition_duration = 200  # Milliseconds for transition
        self.start_time = 0
        self.is_click = False
        self.clicked = False
        self.callback = callback
        self.border_color = border_color
        self.text_color = text_color

    def is_point_inside(self, x, y):
        return self.rect.collidepoint(x, y)

    def update(self):
        mouse_x, mouse_y = pygame.mouse.get_pos()

        if self.is_point_inside(mouse_x, mouse_y):
            if not self.is_hovered:
                self.is_hovered = True
                self.start_time = pygame.time.get_ticks()
        else:
            if self.is_hovered:
                self.is_hovered = False
                self.start_time = pygame.time.get_ticks()
    
        left_button, middle_button, right_button = pygame.mouse.get_pressed()    
       
        if left_button and not self.is_click and self.is_hovered:
            self.is_click = True
            self.clicked = False

        if self.is_click and not left_button:
            self.is_click = False


        if not self.clicked and self.is_click and self.callback is not None:
            self.callback()
            self.clicked = True

        # for event in pygame.event.get():
        #     if event.type == pygame.MOUSEBUTTONDOWN:
        #         print("level 1")
        #         if self.is_hovered and self.callback is not None:
        #             self.callback()


    def draw(self, screen):
        current_time = pygame.time.get_ticks()
        time_elapsed = current_time - self.start_time
        if self.is_hovered:
            size_factor = min(1.0, time_elapsed / self.transition_duration)
        else:
            size_factor = max(0.0, 1.0 - time_elapsed / self.transition_duration)

        current_width = int(self.rect.width + size_factor * 10)
        current_height = int(self.rect.height + size_factor * 10)

        button_rect = pygame.Rect(self.rect.x - (current_width - self.rect.width) // 2,
                                  self.rect.y - (current_height - self.rect.height) // 2,
                                  current_width, current_height)

        border_color = self.border_color
        text_color = self.text_color

        if self.is_hovered:
            color = self.highlight_color
        else:
            color = self.color

        pygame.draw.rect(screen, color, button_rect, 0, 20)
        pygame.draw.rect(screen, border_color, button_rect, 3, 20)

        font = pygame.font.SysFont(None, self.font_size)
        text_surface = font.render(self.text, True, text_color)
        text_rect = text_surface.get_rect(center=button_rect.center)
        screen.blit(text_surface, text_rect)
