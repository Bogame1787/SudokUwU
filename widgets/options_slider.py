import pygame
import sys
from widgets.colors import *
from widgets.button import Button

class OptionsSlider:
    def __init__(self, x, y, width, height, num_levels, default_level=1):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.num_levels = num_levels
        self.current_level = default_level
        self.max_level = num_levels
        self.min_level = 1

        self.increase_button = Button(x + width - 50, y, 110, height, "Increase", font_size=30, color=BLUE, highlight_color=BLUE_HIGHLIGHT, callback=self.increase_difficulty)
        self.decrease_button = Button(x-50, y, 110, height, "Decrease", font_size=30, color=RED, highlight_color=RED_HIGHLIGHT, callback=self.decrease_difficulty)

    def increase_difficulty(self):
        self.current_level += 1
        if self.current_level > self.max_level:
            self.current_level = self.min_level

    def decrease_difficulty(self):
        self.current_level -= 1
        if self.current_level < self.min_level:
            self.current_level = self.max_level

    def update(self):
        self.increase_button.update()
        self.decrease_button.update()

    def draw(self, screen):
        font = pygame.font.SysFont(None, 40)
        level_text = font.render(f"Difficulty Level: {self.current_level}", True, WHITE)
        level_rect = level_text.get_rect(center=(self.x + self.width // 2, self.y + self.height // 2))
        pygame.draw.rect(screen, BLACK, (self.x, self.y, self.width, self.height))
        # pygame.draw.rect(screen, RED, (self.x, self.y, self.width, self.height), 2)
        screen.blit(level_text, level_rect)

        # Draw the increase and decrease buttons
        self.increase_button.draw(screen)
        self.decrease_button.draw(screen)