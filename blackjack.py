import gymnasium as gym
import numpy as np

from cards import Deck
from gymnasium import spaces


class BlackjackHand:
    def __init__(self):
        self.cards = []
        self.has_ace = False

    def add_card(self, card):
        if card[0] == 1:
            self.has_ace = True
        self.cards.append(card)

    def calculate_value(self):
        total = 0
        aces = 0
        for card, suit in self.cards:
            if card == 1:
                aces += 1
                total += 11  # initially consider Ace as 11
            else:
                total += card

        # If total is greater than 21 and there are aces, count aces as 1
        while total > 21 and aces:
            total -= 10
            aces -= 1

        return total

    def is_busted(self):
        return self.calculate_value() > 21

    def is_blackjack(self):
        return len(self.cards) == 2 and self.calculate_value() == 21


class BlackjackEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_decks=1):
        """
        Initializes the environment and the deck.
        :param num_decks: Number of decks to use in the game (default is 1)
        """
        super(BlackjackEnv, self).__init__()

        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Dict({
            'player_hand_value': spaces.Box(low=0, high=31, shape=(1,), dtype=np.int32),
            'dealer_visible_card': spaces.Box(low=0, high=10, shape=(1,), dtype=np.int32)
        })

        # Initialize game components
        self.deck = Deck(num_decks)
        self.player_hand = BlackjackHand()
        self.dealer_hand = BlackjackHand()
        self.bet_placed = False
        self.done = False
        self.reward = 0
        self.double: bool = False

    def reset(self, seed=None, options=None):
        """
        Resets the game state.
        The player must place a bet first. If no bet is placed, the episode ends.
        """
        super().reset(seed=seed)
        if len(self.deck.cards) < 20:
            self.deck.reset()
        self.player_hand = BlackjackHand()
        self.dealer_hand = BlackjackHand()
        self.bet_placed = False
        self.done = False
        self.reward = 0
        self.double = False

        return {"player_hand_value": np.array([0]), "dealer_visible_card": np.array([0])}, {}

    def simulate(self):
        for _ in range(4):
            self.deck.draw_card()

    def step(self, action):
        # should bet_placed be sent in obs?
        if not self.bet_placed:
            if action == 0:
                self.done = True
                self.reward = -0.5 + 0.2 * \
                    (len(self.deck.cards)/(52.0*self.deck.num_decks))
                self.reward = -0.5
                self.simulate()
                return self._get_obs(), self.reward, self.done, False, self._info()

            elif action == 1:
                # Player places a bet, cards are dealt
                self.bet_placed = True
                for _ in range(2):
                    self.player_hand.add_card(self.deck.draw_card())
                    self.dealer_hand.add_card(self.deck.draw_card())
                if self.player_hand.is_blackjack():
                    self.done = True
                    self.reward = 1.5
                return self._get_obs(), self.reward, self.done, False, self._info()
        else:
            if action == 2:
                if not self.double:
                    self.double = True
                    self.reward = 0
                    return self._get_obs(), self.reward, self.done, False, self._info()

            elif action == 3:  # Hit
                self.player_hand.add_card(self.deck.draw_card())
                self.reward = 0
                if self.player_hand.is_busted():
                    self.done = True
                    self.reward = -1*self._reward_multiplier()
                return self._get_obs(), self.reward, self.done, False, self._info()

            elif action == 4:  # Stand
                self._dealer_turn()
                self._determine_winner()
                self.reward = self.reward * self._reward_multiplier()
                return self._get_obs(), self.reward, self.done, False, self._info()

        return self._get_obs(), -10, True, False, {}

    def _info(self):
        card = 0
        if self.bet_placed:
            card = self.dealer_hand.cards[0][0]
        return {
            'double': self.double,
            'player_has_ace': self.player_hand.has_ace,
            'player_hand_value': self.player_hand.calculate_value(),        """
        Returns the observation, which is the player's hand value and the dealer's visible card.
        """
            'dealer_visible_card': card,
        }

    def _reward_multiplier(self):
        if self.double:
            return 2
        return 1

    def render(self, mode='human'):
        """
        Prints out the current game state.
        """
        if self.bet_placed:
            print(
                f"Player's hand: {self.player_hand.cards} (value = {self.player_hand.calculate_value()})")
            if self.done:
                print(
                    f"Dealer's hand: {self.dealer_hand.cards} (value = {self.dealer_hand.calculate_value()})")
            else:
                print(f"Dealer's visible card: {self.dealer_hand.cards[0]}")

        else:
            print(f"Betting Phase")

    def _get_obs(self):
        if not self.bet_placed:
            return {"player_hand_value": np.array(
                [0]), "dealer_visible_card": np.array([0])}
        return {
            'player_hand_value': np.array([self.player_hand.calculate_value()]),
            'dealer_visible_card': np.array([self.dealer_hand.cards[0][0]])
        }

    def _dealer_turn(self):
        while self.dealer_hand.calculate_value() < 17:
            self.dealer_hand.add_card(self.deck.draw_card())

    def _determine_winner(self):
        player_value = self.player_hand.calculate_value()
        dealer_value = self.dealer_hand.calculate_value()

        if self.dealer_hand.is_busted() or player_value > dealer_value:
            self.reward = 1  # Player wins
        elif player_value < dealer_value:
            self.reward = -1  # Player loses
        else:
            self.reward = 0  # Draw
        self.done = True
