from typing import List

import numpy as np

from character_recognition.character_recognizer import LABELS


class NGramObservation:
    def __init__(self, frequency: int, combination: List[int]):
        self.frequency = frequency
        self.combination = combination


class NGramProcessor:
    observations: List[NGramObservation] = []
    total_ngram_count: int = 0
    longest_combination: int = 0
    transitions: np.ndarray = np.zeros(0)

    def __init__(self, ngram_file: str, ngram_length: int):
        """
        :param ngram_file: The file containing n-gram definitions of the shape [frequency [char[0], char[1], ... char[x]].
        :param ngram_length: The length of the n-grams to use. For example, when using a value of 2, all higher order
        n-grams will be reduced to bi-grams.
        """
        assert ngram_length > 1
        self.__ngram_file: str = ngram_file
        self._ngram_length: int = ngram_length
        self.__parse_file()
        self.__reduce_to_size(self._ngram_length)
        self.matrix: np.ndarray = self.__get_matrix()

    def predict_multiple(self, observations: List[List[int]]) -> List[List[int]]:
        """
        Predicts multiple lists of observations in one go. See `predict` for more information.
        """
        return [self.predict(obs) for obs in observations]

    def predict(self, observation: List[int]) -> List[int]:
        """
        Predicts the most likely sequence of states given an observation.

        :param observation: An observation of a sequence of states.
        :return: The most likely sequence of states.
        """
        if self._ngram_length != 2:
            raise ValueError("n-gram size of {} is not supported! Only size 2 is supported currently!"
                             .format(self._ngram_length))

        if len(observation) < 2:
            return observation

        from word_recognition.viterbi import viterbi

        observation: np.ndarray = np.asarray(observation)
        states: np.ndarray = np.asarray(range(len(LABELS)))
        start: np.ndarray = np.asarray(OBSERVED_ACCURACIES)

        uncertainty = 0.15
        for idx in range(start.shape[0]):
            val = start[idx]
            if val > 0.5:
                start[idx] -= uncertainty
            else:
                start[idx] += uncertainty
        emission: np.ndarray = self.matrix

        trans_mat = self.transitions.copy()

        start /= np.sum(start)
        for i in range(trans_mat.shape[0]):
            trans_mat[i, :] /= np.sum(trans_mat[i, :])

        return viterbi(observation, states, start, trans_mat, emission)

    def __parse_file(self):
        with open(self.__ngram_file) as file:
            for line in file.readlines():
                line_data: List[int] = [int(val) for val in line.split()]
                combination = line_data[1:]
                self.observations.append(NGramObservation(line_data[0], combination))
                self.total_ngram_count += line_data[0]
                self.longest_combination = max(self.longest_combination, len(line_data) - 1)

    def __reduce_to_size(self, new_size: int):
        new_observations = []
        new_total_frequency = 0
        for observation in self.observations:
            if len(observation.combination) == new_size:
                self.__potential_duplicate_append(new_observations, observation)
                new_total_frequency += observation.frequency
            elif len(observation.combination) > new_size:
                for idx in range(len(observation.combination) - new_size):
                    new_combination = NGramObservation(observation.frequency,
                                                       observation.combination[idx:idx + new_size])

                    self.__potential_duplicate_append(new_observations, new_combination)
                    new_total_frequency += new_combination.frequency

        self.observations = new_observations
        self.total_ngram_count = new_total_frequency
        self.longest_combination = new_size
        self.__update_probabilities()

    def __potential_duplicate_append(self, obvervations: List[NGramObservation], new_observation: NGramObservation):
        for observation in obvervations:
            if observation.combination == new_observation.combination:
                observation.frequency += new_observation.frequency
                return
        obvervations.append(new_observation)

    def __update_probabilities(self):
        self.transitions = np.zeros(np.repeat(len(LABELS), self._ngram_length), dtype=np.double)
        for observation in self.observations:
            # Update the frequency of transitions[combination[0], combination[1], ... combination[n]].
            current_transition = self.transitions[observation.combination[0]]
            idx: int = 0
            for idx in range(1, self._ngram_length - 1):
                current_transition = current_transition[observation.combination[idx]]
            current_transition[observation.combination[idx + 1]] = observation.frequency

        self.transitions += 1  # Naively avoid 0-probability
        self.transitions /= self.total_ngram_count

    def __get_matrix(self) -> np.ndarray:
        matrix: np.ndarray = np.asarray(MATRIX)
        probabilities: np.ndarray = np.zeros(matrix.shape, dtype=np.double)
        for idx in range(matrix.shape[0]):
            probabilities[idx, :] = matrix[idx, :] / np.sum(matrix[idx, :])
        return probabilities


OBSERVED_ACCURACIES: List[float] = [0.98333333, 0.88611111, 0.96111111, 0.71296296, 0.95833333, 0.80434783, 0.,
                                    0.95833333, 0.94722222, 0.56666667, 0.9122807, 0.41666667, 0.96327684,
                                    0.91388889, 0.96388889, 0.95, 0.89102564, 0.93055556, 0.98333333, 0.54761905,
                                    0.72222222, 0.98611111, 1., 0.96540881, 0.35555556, 0.96944444, 0.98333333]

MATRIX: List[List[int]] = [[1782, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1666, 0, 0, 0, 0, 0, 0, 0, 0, 109, 0, 0, 0, 3, 3, 0, 2, 6, 8, 0, 0, 0, 0, 0, 0, 3],
                           [4, 0, 1778, 0, 0, 0, 0, 1, 0, 3, 0, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [8, 0, 12, 477, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 41, 0, 0],
                           [3, 0, 0, 0, 1758, 0, 0, 25, 0, 0, 0, 0, 0, 0, 8, 1, 0, 1, 1, 0, 0, 0, 0, 3, 0, 0, 0],
                           [4, 0, 0, 0, 0, 596, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 82, 0, 0, 0, 0, 0, 3, 0, 5, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 71, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [27, 0, 1, 0, 16, 0, 0, 1756, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1760, 0, 0, 0, 0, 0, 4, 0, 0, 34, 0, 0, 0, 0, 0, 0, 0, 2, 0],
                           [0, 0, 22, 0, 0, 12, 0, 0, 0, 101, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 40, 0, 0, 0, 0, 0, 0, 0, 0, 1074, 0, 0, 0, 1, 26, 0, 0, 0, 9, 0, 8, 0, 0, 0, 0, 0],
                           [0, 0, 0, 2, 0, 4, 0, 0, 0, 0, 0, 27, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 6, 0, 16, 0, 0],
                           [0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1730, 0, 0, 6, 1, 0, 21, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 43, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1733, 12, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1762, 0, 0, 32, 3, 0, 0, 0, 0, 1, 0, 1, 0],
                           [0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 6, 1775, 0, 0, 1, 5, 0, 8, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 744, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0],
                           [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 22, 0, 1, 1728, 11, 0, 0, 0, 4, 24, 0, 0, 6],
                           [1, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1793, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 13, 0, 0, 0, 0, 0, 0, 2, 0, 25, 0, 0, 0, 6, 7, 0, 0, 0, 169, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 4, 0, 0, 1792, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 438, 0, 0, 0, 0],
                           [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 2, 0, 0, 0, 12, 1570, 0, 0, 0],
                           [0, 0, 4, 332, 0, 3, 0, 0, 0, 5, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 121, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 1776, 0],
                           [0, 0, 1, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1786]]
