import copy
from random import shuffle

from .card import Card
from .player import Player
import collections

from .pygame_renderer import PygameRenderer

ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']
suits = ['Clovers', 'Diamonds', 'Hearts', 'Spades']


def _get_deck(full_deck: bool):
    deck = []
    for i in range(0 if full_deck else 7, 13):
        for j in range(4):
            deck.append(Card(suits[j], ranks[i]))
    return deck


def _chunk(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class Kierki:
    """
    Black lady game. Requires 4 unique player objects.
    """

    def __init__(self, player1: Player, player2: Player, player3: Player, player4: Player, display=True, delay=500,
                 full_deck: bool = True):
        """
        If display is True, the game will open a pygame window and use the delay argument as the delay between moves.
        Otherwise, the game will calculate all moves instantly once started.
        Set the full_deck argument to False to use a smaller deck with cards of rank 9 and above (inclusive).
        """
        self.full_deck = full_deck
        self.players = collections.deque([player1, player2, player3, player4])
        self.state = {
            "hands": self._deal(),
            "discard": {},
            "points": {
                player1: 0,
                player2: 0,
                player3: 0,
                player4: 0
            },
            "old_discards": []
        }
        if display:
            self.renderer = PygameRenderer(delay)
        else:
            self.renderer = None

    def _deal(self) -> dict:
        deck = _get_deck(self.full_deck)
        shuffle(deck)
        hands = dict(
            zip(self.players, _chunk(deck, 13 if self.full_deck else 6))
        )
        return hands

    def _validate(self, player, move: Card) -> bool:
        """
        Validates a move. You can refer to the wikipedia page if this is confusing.
        """

        # obvious
        if move not in self.state["hands"][player]:
            return False

        # the starting player can discard whatever
        if len(self.state["discard"]) == 0:
            return True

        # if the suit is not the same as the suit of the first card and the player could provide it
        if move.suit != iter(self.state["discard"].values()).__next__().suit and \
                list(filter(lambda card: card.suit == iter(self.state["discard"].values()).__next__().suit,
                            self.state["hands"][player])):
            return False

        return True

    def _calc_penalty(self) -> tuple[Player, int]:
        """
        The penalty for losing a round is:
        1 point for each card from the hearts suit
        13 points for the queen of spades
        """
        first_suit = iter(self.state["discard"].values()).__next__().suit
        possible_losers = list(
            filter(lambda player_card: player_card[1].suit == first_suit, self.state["discard"].items()))
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
        """
        Runs the game once (11 deals).
        """
        for _ in range(11):
            points_old = self.state["points"].copy()
            for _ in range(13 if self.full_deck else 6):
                for player in self.players:

                    state_copy = {"hand": copy.deepcopy(self.state["hands"][player]),
                                  "discard": copy.deepcopy(list(self.state["discard"].values()))}
                    move = player.make_move(state_copy)
                    if self._validate(player, move):
                        self.state["hands"][player].remove(move)
                        self.state["discard"][player] = move
                    else:
                        raise Exception("Invalid move")
                    if self.renderer:
                        self.renderer.render(self.state)
                loser, penalty = self._calc_penalty()

                for player in self.players:
                    temp_reward = {player: 0 for player in self.players} | {loser: penalty}
                    discards = {player: self.state["discard"][player] for player in self.players}
                    player.set_temp_reward(
                        discards,
                        temp_reward
                    )

                # the loser starts
                first = self.players.index(loser)
                self.players.rotate(-first)
                self.state["points"][loser] += penalty

                self.state["discard"] = {}
            self.state["hands"] = self._deal()
            for player in self.players:
                points = {player: self.state["points"][player] - points_old[player] for player in self.players}
                player.set_final_reward(points)
        points = self.state["points"]
        self.state["points"] = {player: 0 for player in self.state["points"].keys()}
        return points
