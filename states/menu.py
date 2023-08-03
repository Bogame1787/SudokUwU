import pygame
from states.state import State
from states.sudoku import Sudoku
from widgets.button import Button
from widgets.label import Label
from widgets.options_slider import OptionsSlider
from widgets.colors import *

class MainMenu(State):
    def __init__(self, game):
        State.__init__(self, game)
        self.buttons = []
        self.sliders = []
        self.labels = []
        self.start_button = Button(game.WIDTH // 2 - 50, game.HEIGHT // 2, 100, 50, "PLAY", border_color=GREEN_BORDER, callback=self.start_sudoku)
        self.buttons.append(self.start_button)
        self.label = Label(game.WIDTH // 2, game.HEIGHT // 2 - 100, "SudokUwU", font_size=100, color=WHITE)
        self.labels.append(self.label)
        self.difficulty_selector = OptionsSlider(game.WIDTH // 2 - 200, game.HEIGHT // 2 + 100, 400, 50, 3, default_level=3)
        self.sliders.append(self.difficulty_selector)

    def start_sudoku(self):
        sudoku_state = Sudoku(self.game)
        sudoku_state.enter_state()

    def update(self):
        for button in self.buttons:
            button.update()
    
        for slider in self.sliders:
            slider.update()
            self.game.difficulty - slider.current_level

    def render(self):
        self.game.screen.fill((0,0,0))
        for button in self.buttons:
            button.draw(self.game.screen)

        for label in self.labels:
            label.draw(self.game.screen)
    
        for slider in self.sliders:
            slider.draw(self.game.screen)
       

