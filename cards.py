import random


class Deck:
    def __init__(self, num_decks=1):
        """
        Initializes the deck with the specified number of decks.
        :param num_decks: Number of decks to use (default is 1)
        """
        self.num_decks = num_decks
        self.cards = self._generate_deck()
        random.shuffle(self.cards)
        self.draw_callback = None

    def _generate_deck(self):
        """
        Generates the cards for the specified number of decks.
        A deck contains 52 cards.
        """
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10,
                  10]  # Ace (1), 2-10, Jack/Queen/King (10)
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        single_deck = [(value, suit) for value in values for suit in suits]

        # Repeat the single deck based on num_decks
        full_deck = single_deck * self.num_decks
        return full_deck

    def register_callback(self, draw_callback):
        self.draw_callback = draw_callback

    def draw_card(self):
        """
        Draws a card from the deck.
        If the deck is empty, return None.
        """
        recent_draw = self.cards.pop() if self.cards else None
        if self.draw_callback:
            self.draw_callback(recent_draw)
        return recent_draw

    def reset(self):
        """
        Resets the deck to the original state, reshuffles it.
        """
        self.cards = self._generate_deck()
        random.shuffle(self.cards)
        if self.draw_callback:
            self.draw_callback(None)

    def desk_size(self):
        return len(self.cards)


if __name__ == "__main__":
    # Create a deck with 1 deck by default
    deck = Deck()

    # Loop to draw and print cards from the deck until it's empty
    while True:
        card = deck.draw_card()
        if card is None:
            print("No more cards in the deck!")
            break
        value, suit = card
        print(f"Drawn card: {value} of {suit}")
