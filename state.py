import numpy as np
from cards import Deck


class DeckState:
    def __init__(self, deck: Deck):
        self.n = len(deck.cards)
        self.num_decks = deck.num_decks
        self.reset()
        deck.register_callback(self.drawn)
        # track the number of times each card has been drawn

    def reset(self):
        self.drawn_count = np.ones(10)*4*self.num_decks
        self.drawn_count[9] = 16*self.num_decks
        self.cards_left = self.n

    def drawn(self, card):
        if card == None:
            self.reset()
            return
        self.cards_left -= 1
        self.drawn_count[card[0]-1] -= 1
        if self.drawn_count[card[0]-1] < 0:
            raise Exception(f"invalid Drawn vector {self.drawn}")

    def get_probabilities(self):
        # calculate the probabilities for each card number 1 to 13
        if self.cards_left == 0:
            raise Exception("Zero cards left")
        prob_vector = self.drawn_count/self.cards_left
        return prob_vector

    def state_dim(self):
        return len(self.drawn_count)


if __name__ == "__main__":
    # Create a deck with 1 deck by default
    deck = Deck()
    state = DeckState(deck)

    while (True):
        c = deck.draw_card()
        print(c)
        print(state.get_probabilities())
