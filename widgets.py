import pygame
import sys
from colors import *

pygame.init()

SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
surface = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Sudoky4You Widgets Test")

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

class Button:
    def __init__(self, x, y, width, height, text, font_size=30, color=BLACK):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.highlight_color = GREEN_HIGHLIGHT
        self.text = text
        self.font_size = font_size
        self.is_hovered = False
        self.transition_duration = 200  # Milliseconds for transition
        self.start_time = 0

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
        if self.is_hovered:
            color = self.highlight_color
        else:
            color = self.color

        pygame.draw.rect(screen, color, button_rect, 0, 20)
        pygame.draw.rect(screen, GREEN_BORDER, button_rect, 3, 20)

        font = pygame.font.SysFont(None, self.font_size)
        text_surface = font.render(self.text, True, GREEN)
        text_rect = text_surface.get_rect(center=button_rect.center)
        screen.blit(text_surface, text_rect)

def draw(surface, buttons, labels):
    surface.fill(BLACK)
    
    for button in buttons:
        button.update()
        button.draw(surface)

    for label in labels:
        label.draw(surface)
    
    pygame.display.update()


def main(surface):
    label1 = Label(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 100, "Sudoku4You", font_size=100, color=RED)

    button = Button(SCREEN_WIDTH // 2 - 50, SCREEN_HEIGHT // 2, 100, 50, "PLAY")

    labels = [label1]
    buttons = [button]

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        draw(surface, buttons, labels)

if __name__ == "__main__":
    main(surface)
