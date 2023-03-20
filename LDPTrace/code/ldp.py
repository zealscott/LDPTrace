import numpy as np
import random
import math


class LDPServer:
    def __init__(self, epsilon, d, map_func=None):
        """
        General class of server side
        :param epsilon: privacy budget
        :param d: domain size
        :param map_func: index mapping function
        """
        self.epsilon = epsilon
        self.d = d
        self.map_func = lambda x: x if map_func is None else map_func

        # Sum of updated data
        self.aggregated_data = np.zeros(self.d)
        # Adjusted from aggregated data
        self.adjusted_data = np.zeros(self.d)

        # Number of users
        self.n = 0

    def aggregate(self, data):
        """
        Aggregate users' updated data items
        :param data: real data item updated by user
        """
        raise NotImplementedError('Aggregation on sever not implemented!')

    def adjust(self):
        """
        Adjust aggregated data to get unbiased estimation
        """
        raise NotImplementedError('Adjust on sever not implemented!')

    def initialize(self, epsilon, d, map_func=None):
        self.epsilon = epsilon
        self.d = d
        #self.map_func = lambda x: x if (map_func is None) else map_func
        self.map_func = map_func

        # Sum of updated data
        self.aggregated_data = np.zeros(self.d)
        # Adjusted from aggregated data
        self.adjusted_data = np.zeros(self.d)

        # Number of users
        self.n = 0


class LDPClient:
    def __init__(self, epsilon, d, map_func=None):
        """
        General class of client side
        :param epsilon: privacy budget
        :param d: domain size
        :param map_func: index mapping function
        """
        self.epsilon = epsilon
        self.d = d
        #self.map_func = lambda x: x if (map_func is None) else map_func
        self.map_func = map_func

    def _perturb(self, index):
        """
        Internal method for perturbing real data
        :param index: index of real data item
        """
        raise NotImplementedError('Perturb on client not implemented!')

    def privatise(self, data):
        """
        Public method for privatising real data
        :param data: data item
        """
        raise NotImplementedError('Privatise on sever not implemented!')

    def initialize(self, epsilon, d, map_func=None):
        self.epsilon = epsilon
        self.d = d
        self.map_func = lambda x: x if map_func is None else map_func


class OUEServer(LDPServer):
    def __init__(self, epsilon, d, map_func=None):
        """
        Optimal Unary Encoding of server side
        """
        super(OUEServer, self).__init__(epsilon, d, map_func)

        # Probability of 1=>1
        self.p = 0.5
        # Probability of 0=>1
        self.q = 1 / (math.pow(math.e, self.epsilon) + 1)
        # self.q = 0

    def aggregate(self, data):
        self.aggregated_data += data
        self.n += 1

    def adjust(self) -> np.ndarray:
        # Real data, don't adjust
        if self.epsilon < 0:
            self.adjusted_data = self.aggregated_data
            return self.adjusted_data

        self.adjusted_data = (self.aggregated_data - self.n * self.q) / (self.p - self.q)
        return self.adjusted_data

    def estimate(self, data) -> float:
        """
        Estimate frequency of a specific data item
        :param data: data item
        :return: estimated frequency
        """
        index = self.map_func(data)
        return self.adjusted_data[index]


class OUEClient(LDPClient):
    def __init__(self, epsilon, d, map_func=None):
        """
        Optimal Unary Encoding of server side
        """
        super(OUEClient, self).__init__(epsilon, d, map_func)

        # Probability of 1=>1
        self.p = 0.5
        # Probability of 1=>1
        self.q = 1 / (math.pow(math.e, self.epsilon) + 1)
        # self.q = 0

    def _perturb(self, index):
        # Remember that p is the probability for 1=>1;
        # And q is the probability for 0=>1

        # Update real data
        if self.epsilon < 0:
            perturbed_data = np.zeros(self.d)
            perturbed_data[index] = 1
            return perturbed_data

        # If y=0, Prob(y'=1)=q, Prob(y'=0)=1-q
        perturbed_data = np.random.choice([1, 0], size=self.d, p=[self.q, 1-self.q])

        # If y=1, Prob(y'=0)=p
        if random.random() < self.p:
            perturbed_data[index] = 1
        return perturbed_data

    def privatise(self, data):
        index = self.map_func(data)
        return self._perturb(index)
