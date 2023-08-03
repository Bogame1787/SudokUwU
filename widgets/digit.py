import pygame
from widgets.button import Button
from widgets.colors import *


class Digit:
    def __init__(self, x, y, radius, digit, font_size=30, color=(40,55,73)):
        self.x = x
        self.y = y
        self.radius = radius
        self.digit = str(digit)
        self.font_size = font_size
        self.color = color
        self.highlight_color = (60, 76, 93)
        self.is_hovered = False
        self.transition_duration = 100  # Milliseconds for transition
        self.start_time = 0
        self.is_click = False
        self.clicked = False
        self.callback = self.callback_fn

    def callback_fn(self):
        print(self.digit)

    def is_point_inside(self, x, y):
        distance = ((x - self.x) ** 2 + (y - self.y) ** 2) ** 0.5
        return distance <= self.radius

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
            self.callback_fn()
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
            size_factor = min(0.5, time_elapsed / self.transition_duration)
        else:
            size_factor = max(0.0, 0.5 - time_elapsed / self.transition_duration)

        current_radius = int(self.radius + size_factor * 10)

        if self.is_hovered:
            color = self.highlight_color
        else:
            color = self.color
        pygame.draw.circle(screen, color, (self.x, self.y), current_radius)
        pygame.draw.circle(screen, BLACK, (self.x, self.y), current_radius, 3)

        font = pygame.font.SysFont(None, self.font_size)
        text_surface = font.render(self.digit, True, GREEN)
        text_rect = text_surface.get_rect(center=(self.x, self.y))
        screen.blit(text_surface, text_rect)
