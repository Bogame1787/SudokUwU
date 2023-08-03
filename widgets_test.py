import pygame
import sys
from widgets.colors import *
from widgets.label import Label
from widgets.button import Button
from widgets.options_slider import OptionsSlider
from widgets.digit import Digit

pygame.init()

SCREEN_WIDTH, SCREEN_HEIGHT = 1000, 600
surface = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Sudoky4You Widgets Test")

def draw(surface, buttons, labels, difficulty_bars, digits):
    surface.fill(BLACK)
    
    for button in buttons:
        button.update()
        button.draw(surface)

    for label in labels:
        label.draw(surface)

    for difficulty_bar in difficulty_bars:
        difficulty_bar.update()
        difficulty_bar.draw(surface)

    for digit in digits:
        digit.update()
        digit.draw(surface)
    
    pygame.display.update()

def on_button_click():
    print("Button Clicked!")

def main(surface):
    label1 = Label(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 100, "Sudoku4You", font_size=100, color=WHITE)
    num_levels = 3
    difficulty_selector = OptionsSlider(300, 400, 400, 50, num_levels, default_level=3)
    button = Button(SCREEN_WIDTH // 2 - 50, SCREEN_HEIGHT // 2, 100, 50, "PLAY", border_color=GREEN_BORDER, callback=on_button_click)
    digit = Digit(300, 100, 25, 9, font_size=35)
    labels = [label1]
    buttons = [button]
    difficulty_bars = [difficulty_selector]
    digits = [digit]
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        draw(surface, buttons, labels, difficulty_bars, digits)
        # print(difficulty_selector.current_level)

if __name__ == "__main__":
    main(surface)
