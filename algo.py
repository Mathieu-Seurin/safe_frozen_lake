from copy import copy
import numpy as np


class OptimalAgent(object):

    def __init__(self):

        # self.state_to_best_action = {
        #     0 : 1,
        #     9 : 2,
        #     10 : 1,
        #     13 : 2,
        #     14 : 2,
        #     1 : 0
        # }

        self.state_to_best_action = [
            0, 3, 3, 3, 0,-1, 0,-1, 3, 1, 0,  -1, -1, 2, 1
        #   0  1  2  3  4  5  6  7  8  9  10  11  12 13 14
        ]

        self.current_eps = 0

    def choose_action_eps_greedy(self, s):
        s = s['state']
        return self.state_to_best_action[s]

    def choose_action_greedy(self, s):
        s = s['state']
        return self.state_to_best_action[s]

    def optimize(self, state, a, next_state, r):
        pass

    def test(self):
        pass

    def train(self):
        pass

class TabQLearning(object):
    def __init__(self, env_size, n_action, gamma, lr, expected_exploration_steps):

        self.n_action = n_action
        self.env_size = env_size
        self.q_tab = np.zeros((env_size, n_action))

        self.lr = lr
        self.gamma = gamma

        # Exploration parameters
        self.expected_exploration_steps = expected_exploration_steps
        self.n_step_eps = 0
        self.minimum_epsilon = 0.05
        self.epsilon_init = 1
        self.current_eps = self.epsilon_init

    def choose_action_eps_greedy(self, s):

        if np.random.random() < self.current_eps:
            a = np.random.randint(self.n_action)
        else:
            s = s['state']
            a = np.argmax(self.q_tab[s])

        self.current_eps = max(self.minimum_epsilon,
                               self.epsilon_init * np.exp(- 2.5 * self.n_step_eps / self.expected_exploration_steps))
        self.n_step_eps += 1

        return a

    def choose_action_greedy(self, s):
        s = s['state']
        a = np.argmax(self.q_tab[s])
        return a

    def optimize(self, state, a, next_state, r):
        state, next_state = state['state'], next_state['state']
        self.q_tab[state, a] = (1 - self.lr) * self.q_tab[state, a] + self.lr * (
                    r + self.gamma * np.max(self.q_tab[next_state]))


class TabQLearningControlerFeedback(object):
    def __init__(self, env_size, n_action, gamma, lr, expected_exploration_steps, margin):

        self.n_action = n_action
        self.env_size = env_size
        self.q_tab = np.zeros((env_size, n_action))

        self.lr = lr
        self.gamma = gamma
        self.margin = margin

        # Exploration parameters
        self.expected_exploration_steps = expected_exploration_steps
        self.n_step_eps = 0
        self.minimum_epsilon = 0.05
        self.epsilon_init = 1
        self.current_eps = self.epsilon_init

    def choose_action_greedy(self, s):
        s = s['state']
        a = np.argmax(self.q_tab[s])
        return a

    def choose_action_eps_greedy(self, s):

        s = s['state']

        if np.random.random() < self.current_eps:
            a = np.random.randint(self.n_action)
        else:
            a = np.argmax(self.q_tab[s])

        self.current_eps = max(self.minimum_epsilon,
                               self.epsilon_init * np.exp(- 2.5 * self.n_step_eps / self.expected_exploration_steps))
        self.n_step_eps += 1

        return a

    def optimize(self, state, action, next_state, r):

        state = state['state']

        gave_feedback = next_state['gave_feedback']
        next_state = next_state['state']

        if gave_feedback:
            bad_action = action
            q_with_margin = [self.q_tab[state, a] - self.margin_loss(bad_action, a) for a in range(self.n_action)]
            self.q_tab[state, action] = (1 - self.lr) * self.q_tab[state, action] + self.lr * min(q_with_margin)
        else:
            self.q_tab[state, action] = (1 - self.lr) * self.q_tab[state, action] + self.lr * (
                        r + self.gamma * np.max(self.q_tab[next_state]))


    def margin_loss(self, bad_action, a):
        if bad_action == a:
            return 0
        else:
            return self.margin