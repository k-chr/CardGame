
class Card:
    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank

    def __str__(self):
        return self.rank + " of " + self.suit

    def __hash__(self):
        return hash(self.suit + self.rank)

    def __eq__(self, other):
        return self.suit == other.suit and self.rank == other.rank