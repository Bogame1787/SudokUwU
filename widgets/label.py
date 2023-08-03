import pygame
import sys
from widgets.colors import *

class Label:
    def __init__(self, x, y, text, font_size=30, color=BLACK):
        self.x = x
        self.y = y
        self.text = text
        self.font_size = font_size
        self.color = color

    def draw(self, screen):
        font = pygame.font.SysFont(None, self.font_size)
        text_surface = font.render(self.text, True, self.color)
        text_rect = text_surface.get_rect(center=(self.x, self.y))
        screen.blit(text_surface, text_rect)