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

    @abstractmethod
    def set_temp_reward(self, discarded_cards: dict, point_deltas: dict):
        """
        After four cards get played, every player will receive a dict with:
        - 'discarded_cards': dict of these discarded cards by each player (ordered!)
        - 'point_deltas': dict of points received in this round by each player
        """
        pass

    @abstractmethod
    def set_final_reward(self, points: dict):
        """
        After all cards have been played, every player will receive a dict with
        points received in this round by each player. A game consists of eleven such full rounds.
        """
        pass
