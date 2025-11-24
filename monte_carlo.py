import math
import numpy as np

rng = np.random.default_rng()

class StockPath:
    def __init__(self, count=10**6, spot=100, rate=0.05, divid=0.03, vol=0.10):
        self.count = count
        self.spot = spot
        self.rate = rate
        self.divid = divid
        self.vol = vol

    def geom_brownian(self, time=1, steps=1, rand=[]):
        if not len(rand):
            rand = rng.standard_normal((steps, self.count))
        S_t = np.ones((steps+1,self.count))*self.spot
        if not steps-1:
            S_t[-1] = [self.spot*math.exp((self.rate - self.divid)*time - \
                0.5*(self.vol**2)*time + self.vol*math.sqrt(time)*x) \
                for x in rand[-1]]
        else:
            for i in range(steps):
                S_t[i+1] = S_t[i]*(1 + (self.rate - self.divid)*time/steps + \
                    self.vol*math.sqrt(time/steps)*rand[i])
        return S_t

class EuropeanOption:
    def __init__(self, stock, expiry, payoff):
        self.stock = stock
        self.expiry = expiry
        self.payoff = lambda S: np.array([payoff(x) for x in S])

    def _random_seed(func):
        def wrapper(self, steps=1, rand=[]):
            if not len(rand):
                rand = rng.standard_normal((steps, self.stock.count))
            return func(self, steps, rand)
        return wrapper

    @_random_seed
    def price(self, steps, rand):
        S_t = self.stock.geom_brownian(self.expiry, steps, rand)[-1]
        return math.exp(-self.stock.rate*self.expiry) * \
            np.mean(self.payoff(S_t))

    @_random_seed
    def delta(self, steps, rand):
        epsilon = 0.01
        S_1 = self.stock.geom_brownian(self.expiry, steps, rand)[-1]
        self.stock.spot += epsilon
        S_2 = self.stock.geom_brownian(self.expiry, steps, rand)[-1]
        self.stock.spot -= epsilon
        return math.exp(-self.stock.rate*self.expiry) * \
            np.mean(self.payoff(S_2) - self.payoff(S_1))/epsilon

    @_random_seed
    def gamma(self, steps, rand):
        epsilon = 0.01
        delta_1 = self.delta(steps, rand)
        self.stock.spot += epsilon
        delta_2 = self.delta(steps, rand)
        self.stock.spot -= epsilon
        return (delta_2 - delta_1)/epsilon

    @_random_seed
    def vega(self, steps, rand):
        epsilon = 0.001
        S_1 = self.stock.geom_brownian(self.expiry, steps, rand)[-1]
        self.stock.vol += epsilon
        S_2 = self.stock.geom_brownian(self.expiry, steps, rand)[-1]
        self.stock.vol -= epsilon
        return math.exp(-self.stock.rate*self.expiry) * \
            np.mean(self.payoff(S_2) - self.payoff(S_1))/epsilon

    @_random_seed
    def rho(self, steps, rand):
        epsilon = 0.001
        S_1 = self.stock.geom_brownian(self.expiry, steps, rand)[-1]
        self.stock.rate += epsilon
        S_2 = self.stock.geom_brownian(self.expiry, steps, rand)[-1]
        self.stock.rate -= epsilon
        return math.exp(-self.stock.rate*self.expiry) * \
            np.mean(self.payoff(S_2) - self.payoff(S_1))/epsilon

    @_random_seed
    def theta(self, steps, rand):
        epsilon = 0.01
        S_1 = self.stock.geom_brownian(self.expiry, steps, rand)[-1]
        S_2 = self.stock.geom_brownian(self.expiry-epsilon, steps, rand)[-1]
        return math.exp(-self.stock.rate*self.expiry) * \
            np.mean(self.payoff(S_2) - self.payoff(S_1))/epsilon
