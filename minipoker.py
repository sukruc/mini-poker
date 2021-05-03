"""Andrew Moore's Mini Poker Game.

- A is dealt a black (0) or red (1) card randomly with 50% probability each.
- A may resign (0) if red: -20 for A
    - else A holds:
        - B resigns (0): +10 for A
        - B sees (1):
            - If red: -40 for A
            - If black: +30 for A

Define a strategy for Player A that can yield positive average reward at the end of
10,000 games.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, preprocessing, model_selection, feature_selection, metrics, decomposition, cluster, pipeline
import seaborn as sns


class Player:
    """Base class for mini poker players."""
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, arg):
        return self.strategy(arg)

    def strategy(self, arg):
        return None

    def observe(self, reward):
        pass


class PlayerA(Player):
    # TODO: If your player requires parameters, add __init__ method
    # Otherwise, skip this section and proceed to `strategy`
    def __init__(self, *args, **kwargs):
        self.counter = 0
        self.rewards = []


    def strategy(self, card):
        """Define A's strategy.

        Arguments:
        ------------------
        card : int, {0, 1}
            0 is black and 1 is red

        Returns:
        ------------------
        action: int, {0, 1}
            0 resign and 1 hold
        """
        # TODO: Modify Player A's strategy

#         return 1  # her zaman oyna

        # Example strategy: always hold if black, always resign if red
        zar = [0,1]
        olasılık = 0.4
        p = [olasılık, 1-olasılık]
        if card == 0:
             return 1 # hold
        else:
              return np.random.choice(zar, size=1, p=p)

    def observe(self, reward, action_b):
        self.counter += 1
        self.rewards.append(reward)
        pass


class PlayerB(Player):
    def __init__(self, alpha=0.02, epsilon=0.9, epsilon_decay=0.9, alpha_decay=0.99):
        self.Qsa = [0., 0.]
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.alpha_decay = alpha_decay
        self._Q_arr = []

    def strategy(self, action_a):
        """Defines Player B's strategy.

        Arguments:
        ------------------
        action_a : int, {0, 1}
            Player A's action. Actually, if it's Player B's turn, A must have
            hold so this argument is just there for compatibility.

        Returns:
        ------------------
        action_b : int, {0, 1}
            Player B's action. 0 for resign and 1 for see
        """
        if np.random.random() < self.epsilon:
            action = np.random.choice([0, 1])
        else:
            action = np.argmax(self.Qsa)
        self.epsilon *= self.epsilon_decay
        self.last_action = action
        return self.last_action

    def observe(self, reward):
        """Observe reward and redefine strategy."""
        self.Qsa[self.last_action] = self.Qsa[self.last_action] * (1. - self.alpha) + reward * self.alpha
        self._Q_arr.append(self.Qsa.copy())
        self.alpha *= self.alpha_decay


def play_hand(strategy_a: callable, strategy_b: callable) -> int:
    """Play one hand of mini-poker.

    Player A's actions: {0: resign, 1: hold}
    Player B's actions: {0: resign, 1: see}
    Cards: {0: black, 1: red}

    Arguments:
    ---------------
    strategy_a: callable
        A function that takes an integer and returns an integer
    strategy_b: callable
        A function that takes A's action as an integer and returns an integer

    Returns:
    ---------------
    reward_a : int
        Reward for player A

    Notes:
    ---------------
    Mini-poker is adapted from Andrew Moore's slides and game is defined as
    follows:

    - A is dealt a black (0) or red (1) card randomly with 50% probability each.
    - A may resign (0) if red: -20 for A
        - else A holds:
            - B resigns (0): +10 for A
            - B sees (1):
                - If red: -40 for A
                - If black: +30 for A
    """

    B_sees_black = 30
    B_sees_red = -40
    B_resigns = 10
    A_resigns = -20

    card = np.random.choice([0, 1])
    action_a = strategy_a(card)
    action_b = strategy_b(action_a)
    if not action_a:
        reward = A_resigns
    else:
        if not action_b:
            reward = B_resigns
        else:
            if card:
                reward = B_sees_red
            else:
                reward = B_sees_black

    strategy_b.observe(-reward)
    strategy_a.observe(reward, action_b)
    return reward

if __name__ == '__main__':
    N = 200000
    b = PlayerB(0.1, 0.999, 0.99, 0.99)
    a = PlayerA()
    rews = []
    for i in range(N):
        rew = play_hand(a, b)
        rews.append(rew)

    ortalama_kazanc = np.mean(rews)
    print('Alpha decaying:')
    if ortalama_kazanc >= 0.:
        print("Congrats, you won.")
    else:
        print("You lost.")
    print(f'Average reward at the end of {N} games:', ortalama_kazanc)

    b = PlayerB(0.001, 0, 0.9999, 1.0)
    a = PlayerA()
    rews = []
    for i in range(N):
        rew = play_hand(a, b)
        rews.append(rew)

    ortalama_kazanc = np.mean(rews)
    print('Alpha not decaying:')
    if ortalama_kazanc >= 0.:
        print("Congrats, you won.")
    else:
        print("You lost.")
    print(f'Average reward at the end of {N} games:', ortalama_kazanc)
