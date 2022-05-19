import copy
from random import shuffle

from .card import Card
from .player import Player
import collections

from .pygame_renderer import PygameRenderer

ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']
suits = ['Clovers', 'Diamonds', 'Hearts', 'Spades']


def _get_deck():
    deck = []
    for i in range(0, 13):
        for j in range(4):
            deck.append(Card(suits[j], ranks[i]))
    return deck


def _chunk(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def rotate(l, n):
    return l[n:] + l[:n]


class Kierki:
    def __init__(self, player1: Player, player2: Player, player3: Player, player4: Player, display=True):
        self.players = collections.deque([player1, player2, player3, player4])
        self.state = {
            "hands": self._deal(),
            "discard": {},
            "points": {
                player1: 0,
                player2: 0,
                player3: 0,
                player4: 0
            }
        }
        if display:
            self.renderer = PygameRenderer(800, 800, 100)
        else:
            self.renderer = None


    def _deal(self) -> dict:
        deck = _get_deck()
        shuffle(deck)
        hands = dict(
            zip(self.players, _chunk(deck, 13))
        )
        return hands

    def _validate(self, player, move: Card) -> bool:
        # obvious
        if move not in self.state["hands"][player]:
            return False

        if len(self.state["discard"]) == 0:
            return True

        # if the suit is not the same as the suit of the first card and the player could provide it
        if move.suit != iter(self.state["discard"].values()).__next__().suit and \
                list(filter(lambda card: card.suit == iter(self.state["discard"].values()).__next__().suit, self.state["hands"][player])):
            return False

        return True

    def _calc_penalty(self) -> tuple[Player, int]:
        first_suit = iter(self.state["discard"].values()).__next__().suit
        possible_losers = list(filter(lambda player_card: player_card[1].suit == first_suit, self.state["discard"].items()))
        possible_losers.sort(key=lambda player_card: ranks.index(player_card[1].rank))
        loser = possible_losers[-1][0]
        penalty = 0
        for card in self.state["discard"].values():
            if card.suit == 'Hearts':
                penalty += 1
            if card.suit == 'Spades' and card.rank == 'Queen':
                penalty += 13
        return loser, penalty

    def start(self):
        for _ in range(11):
            for _ in range(13):
                self.players.rotate(1)
                for player in self.players:
                    state_copy = {"hand": copy.deepcopy(self.state["hands"][player]), "discard": copy.deepcopy(self.state["discard"])}
                    move = player.make_move(state_copy)
                    if self._validate(player, move):
                        self.state["hands"][player].remove(move)
                        self.state["discard"][player] = move
                    else:
                        raise Exception("Invalid move")
                    if self.renderer:
                        self.renderer.render(self.state)
                loser, penalty = self._calc_penalty()
                self.state["points"][loser] += penalty

                self.state["discard"].clear()
                if not iter(self.state["hands"].values()).__next__():
                    self.state["hands"] = self._deal()

        return self.state["points"]
