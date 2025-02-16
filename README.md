# Blackjack Reinforcement Learning

## Overview

This project implements a Blackjack environment using Gymnasium and a PPO agent using Stable Baselines 3.
You can customize the number of decks in the game by changing the `--deck_size` argument.
Motivation: if we include seencards history in the state, can we imporve the odds.

# Base Agent

State: 
* Probabilities of each card in the deck.
* Entropy of the deck.
* Player's hand value.
* Dealer's visible card.

Actions:
* No Bet
* Bet
* Hit
* Stand
* Double

Rewards:
* Pealize for invalid actions.
* Penalize propotional to entory. To encourage the agent to bet more as we get more information (from seen cards)

