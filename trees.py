import math
import numpy as np

rng = np.random.default_rng()

class Stock:
    def __init__(self, spot=100, rate=0.05, divid=0.03, vol=0.10):
        self.spot = spot
        self.rate = rate
        self.divid = divid
        self.vol = vol

    def geom_brownian(self, time=1, steps=1000):
        self.up = math.exp(self.vol * math.sqrt(time / steps))
        self.down = 1/self.up
        self.pr = (math.exp(self.rate * time / steps) - self.down) / \
            (self.up - self.down)
        S = [[self.spot]*(i+1) for i in range(steps+1)]
        for i in range(1, steps+1):
            for j in range(i+1):
                S[i][j] *= self.up**(i-j) * self.down**j
        return S

class EuropeanOption:
    def __init__(self, stock, expiry, payoff):
        self.stock = stock
        self.expiry = expiry
        self.payoff = lambda S: [payoff(x) for x in S]

    def price(self, steps=1000):
        S = self.stock.geom_brownian(self.expiry, steps)
        value = [self.payoff(x) for x in S]
        for i in reversed(range(steps)):
            for j in range(i+1):
                value[i][j] = math.exp(-self.stock.rate * self.expiry / \
                    steps) * (self.stock.pr * value[i+1][j] + \
                    (1 - self.stock.pr) * value[i+1][j+1])
        return value[0][0]

class AmericanOption:
    def __init__(self, stock, expiry, payoff):
        self.stock = stock
        self.expiry = expiry
        self.payoff = lambda S: [payoff(x) for x in S]

    def price(self, steps=1000):
        S = self.stock.geom_brownian(self.expiry, steps)
        value = [self.payoff(x) for x in S]
        for i in reversed(range(steps)):
            for j in range(i+1):
                value[i][j] = max(value[i][j], math.exp(-self.stock.rate * \
                    self.expiry / steps) * (self.stock.pr * value[i+1][j] + \
                    (1 - self.stock.pr) * value[i+1][j+1]))
        return value[0][0]

class BermudanOption:
    def __init__(self, stock, times, payoff):
        self.stock = stock
        self.times = times
        self.payoff = lambda S: [payoff(x) for x in S]

    def price(self, steps=100):
        total_steps = steps * len(self.times)
        dt = self.times[-1] / total_steps
        S = self.stock.geom_brownian(self.times[-1], total_steps)
        value = [self.payoff(x) for x in S]
        for i in reversed(range(total_steps)):
            if i*dt in self.times:
                for j in range(i+1):
                    value[i][j] = max(value[i][j], math.exp(-self.stock.rate * \
                        dt) * (self.stock.pr * value[i+1][j] + \
                        (1 - self.stock.pr) * value[i+1][j+1]))
            else:
                for j in range(i+1):
                    value[i][j] = math.exp(-self.stock.rate * dt) * \
                        (self.stock.pr * value[i+1][j] + (1 - self.stock.pr) * \
                        value[i+1][j+1])
        return value[0][0]
