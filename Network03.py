class Neuron:
    weights = []
    prevWeights = []

    def normalize(self, signals):
        pass

    def strength(self, signals):
        strength = 0
        for signal in signals:
            strength += abs(signal)
        return strength
