from typing import List

import numpy as np


def viterbi(observations: np.ndarray, states: np.ndarray, initial_probs: np.ndarray, transition: np.ndarray,
            emission: np.ndarray):
    """
    Finds the most likely sequence of states.

    :param observations: The observed sequence of states.
    :param states: The list of possible states.
    :param initial_probs: The initial probabilities.
    :param transition: The transition matrix. Currently only supports a 2d matrix (e.g. for bi-grams). Describes the
    probability to transition from one state to another.
    :param emission: The emission matrix. This describes the probability of observing a state given a state. E.g. a
    confusion matrix.
    :return: The most likely sequence of states.
    """
    path_probs: np.ndarray = np.empty(shape=(len(observations), len(states)), dtype=np.double)
    path: np.ndarray = np.zeros(shape=(len(observations), len(states)), dtype=int)

    path[0, :] = -1
    path_probs[0, :] = initial_probs[:] * emission[:, observations[0]]

    for idx_obs in range(1, len(observations)):
        for idx_state in range(len(states)):
            max_transition_prob: float = path_probs[idx_obs - 1, 0] * transition[states[0], states[idx_state]]

            prev_st_selected: int = 0
            for prev_st in range(1, len(states)):
                tr_prob: float = path_probs[idx_obs - 1, prev_st] * transition[states[prev_st], states[idx_state]]
                if tr_prob > max_transition_prob:
                    max_transition_prob: float = tr_prob
                    prev_st_selected: int = prev_st

            max_prob: float = max_transition_prob * emission[idx_state, observations[idx_obs]]
            path[idx_obs][idx_state] = prev_st_selected
            path_probs[idx_obs][idx_state] = max_prob

    opt: List[int] = []
    max_prob: float = 0.0
    best_st: int = -1

    for idx in range(len(states)):
        if path_probs[-1, idx] > max_prob:
            max_prob: float = path_probs[-1, idx]
            best_st: int = idx
    opt.append(best_st)
    previous: int = best_st

    for idx in range(path.shape[0] - 2, -1, -1):
        opt.insert(0, path[idx + 1, previous])
        previous: int = path[idx + 1, previous]

    return opt
