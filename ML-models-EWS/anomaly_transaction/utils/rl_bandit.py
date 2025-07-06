import numpy as np
from collections import defaultdict

class ContextualBandit:
    def __init__(self, n_arms, context_dim):
        self.n_arms = n_arms  # Number of models
        self.context_dim = context_dim  # Feature dimension
        self.weights = [np.zeros(context_dim) for _ in range(n_arms)]
        self.counts = [0] * n_arms
        self.alpha = 0.1  # Learning rate

    def select_arm(self, context):
        scores = [np.dot(self.weights[i], context) for i in range(self.n_arms)]
        return np.argmax(scores)

    def update(self, arm, context, reward):
        self.counts[arm] += 1
        self.weights[arm] += self.alpha * reward * context