from abc import ABC, abstractmethod
from .card import Card


class Player(ABC):
    @abstractmethod
    def make_move(self, game_state: dict) -> Card:
        """
        The player will receive a dict with:
        - 'hand': list of held cards
        - 'discard': list of discarded cards in this round
        - 'old_discards': list of discarded cards, round by round (list of lists of four cards)
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass
