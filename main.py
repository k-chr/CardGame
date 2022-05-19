# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import random

from Kierki import Kierki, Player, Card


class RandomPlayer(Player):

    def make_move(self, game_state: dict) -> Card:
        if not game_state["discard"]:
            return random.choice(game_state["hand"])
        else:
            options = list(filter(lambda card: card.suit == list(game_state["discard"].values())[0].suit, game_state["hand"]))
            if len(options) > 0:
                return random.choice(options)
            else:
                return random.choice(game_state["hand"])


def main():
    print(Kierki(RandomPlayer(), RandomPlayer(), RandomPlayer(), RandomPlayer()).start())


if __name__ == '__main__':
    main()
