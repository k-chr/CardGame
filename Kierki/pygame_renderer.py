import itertools

import pygame
import os
from .card import Card
from .player import Player
from pathlib import Path

filename_codes = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 'Jack': 11, 'Queen': 12,
                  'King': 13, 'Ace': 1}


def card_image(card: Card):
    return f"card_{filename_codes[card.rank]}_{card.suit[:-1]}".lower()


class PygameRenderer:
    def __init__(self, width, height, delay):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        self.color = pygame.color.Color("white")
        self.player_width = height * 0.8
        self.player_height = 84
        self.delay = delay

        self.player_rects = [pygame.Rect(0.1 * width, 0.8 * height, self.player_width, self.player_height),
                             pygame.Rect(0.8 * width, 0.1 * height, self.player_height, self.player_width),
                             pygame.Rect(0.1 * width, 0.1 * height, self.player_width, self.player_height),
                             pygame.Rect(0.1 * width, 0.1 * height, self.player_height, self.player_width)]

        self.cards: dict[str, pygame.Surface] = {}

        for path in Path('.').rglob('*.png'):
            self.cards[os.path.splitext(os.path.basename(path))[0]] = pygame.image.load(path)

        self.card_width = self.player_width // 13
        self.card_height = self.player_height

    def render(self, state: dict):
        self.screen.fill(self.color)
        for player_index, player in enumerate(state["points"].keys()):
            surface = pygame.Surface((self.player_width, self.player_height))
            surface.get_rect().center = self.player_rects[player_index].center
            surface.fill(self.color)
            for card_index, card in enumerate(state["hands"][player]):
                surface.blit(self.cards[card_image(card)],
                             (card_index * self.card_width, 0, self.card_width, self.card_height))
            self.screen.blit(pygame.transform.rotate(surface, 90 * player_index), self.player_rects[player_index])

        pygame.display.flip()
        pygame.time.wait(self.delay)
        pygame.event.get()
