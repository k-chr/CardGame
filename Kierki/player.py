from abc import ABC, abstractmethod
from .card import Card


class Player(ABC):
    @abstractmethod
    def make_move(self, game_state: dict) -> Card:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass