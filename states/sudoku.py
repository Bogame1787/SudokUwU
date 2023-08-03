import pygame
import os
from states.state import State
from widgets.digit import Digit

class Sudoku(State):
    def __init__(self, game):
        State.__init__(self, game)
        self.digits = self.init_digits()
        self.sudoku_surface_size = 500
        self.sudoku_surface = pygame.Surface((self.sudoku_surface_size, self.sudoku_surface_size))

    def init_digits(self):
        x = self.game.WIDTH - 200 + 30 + 25
        y = 100
        digits = []
        for number in range(1, 10):
            digits.append(Digit(x, y, 25, number, 35))
            y+= 90
            
            if number == 5:
                x += 90
                y = 100
        
        return digits
    
    def draw_line_round_corners_polygon(self, p1, p2, c, w, surface):
        p1v = pygame.math.Vector2(p1)
        p2v = pygame.math.Vector2(p2)
        lv = (p2v - p1v).normalize()
        lnv = pygame.math.Vector2(-lv.y, lv.x) * w // 2
        pts = [p1v + lnv, p2v + lnv, p2v - lnv, p1v - lnv]
        pygame.draw.polygon(surface, c, pts)
        pygame.draw.circle(surface, c, p1, round(w / 2))
        pygame.draw.circle(surface, c, p2, round(w / 2))

    def drawgrid(self, rows, cols, surface):
        gapx = (self.sudoku_surface_size)//cols
        gapy = (self.sudoku_surface_size)//rows

        x = 0
        y = 0

        for i in range(rows):
            if(i == rows - 1):
                break
            y += gapy
            self.draw_line_round_corners_polygon((0, y), (self.sudoku_surface_size, y), (255, 0, 0), 5, surface)

        for i in range(cols):
            if(i == rows - 1):
                break
            x += gapx
            self.draw_line_round_corners_polygon((x, 0), (x, self.game.HEIGHT), (255, 0, 0), 5, surface)

    def update(self):
        if not self.digits:
            return
        for digit in self.digits:
            digit.update()

    def render(self):
        self.game.screen.fill((0,0,0))
        self.sudoku_surface.fill((100,100,100))
        self.drawgrid(3, 3, self.sudoku_surface)
        self.game.screen.blit(self.sudoku_surface, (50, 50))
        
        if not self.digits:
            return
        for digit in self.digits:
            digit.draw(self.game.screen)


    