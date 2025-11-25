import math
import numpy as np
from scipy.stats import norm

forward_contract = lambda stock, expiry, strike: \
    stock.spot * math.exp(-stock.divid * expiry) - strike * \
    math.exp(-stock.rate * expiry)

_gaussian = lambda x: math.exp(-0.5 * (x**2))/math.sqrt(2 * math.pi)

def _normal_limits(func):
    def wrapper(self):
        d_1 = (math.log(self.stock.spot / self.strike) + (self.stock.rate - \
            self.stock.divid + 0.5 * (self.stock.vol**2)) * self.expiry)/ \
            (self.stock.vol * math.sqrt(self.expiry))
        d_2 = d_1 - self.stock.vol * math.sqrt(self.expiry)
        return func(self, d_1, d_2)
    return wrapper

class VanillaCall:
    def __init__(self, stock, expiry, strike):
        self.stock = stock
        self.strike = strike
        self.expiry = expiry

    price = _normal_limits(lambda self, d_1, d_2: \
        self.stock.spot * math.exp(-self.stock.divid * self.expiry) * \
        norm.cdf(d_1) - self.strike * math.exp(-self.stock.rate * \
        self.expiry) * norm.cdf(d_2))

    delta = _normal_limits(lambda self, d_1, d_2: \
        math.exp(-self.stock.divid * self.expiry) * norm.cdf(d_1))

    gamma = _normal_limits(lambda self, d_1, d_2: \
        (math.exp(-self.stock.divid * self.expiry) * _gaussian(d_1)) / \
        (self.stock.spot * self.stock.vol * math.sqrt(self.expiry)))

    vega = _normal_limits(lambda self, d_1, d_2: \
        self.stock.spot * math.exp(-self.stock.divid * self.expiry) * \
        math.sqrt(self.expiry) * _gaussian(d_1))

    rho = _normal_limits(lambda self, d_1, d_2: \
        self.strike * math.exp(-self.stock.rate * self.expiry) * \
        self.expiry * norm.cdf(d_2))

    theta = _normal_limits(lambda self, d_1, d_2: \
        self.stock.spot * math.exp(-self.stock.divid * self.expiry) * \
        self.stock.divid * norm.cdf(d_1) - self.strike * \
        math.exp(-self.stock.rate * self.expiry) * self.stock.rate * \
        norm.cdf(d_2) - self.stock.spot * math.exp(-self.stock.divid * \
        self.expiry) * (self.stock.vol / (2 * math.sqrt(self.expiry))) * \
        _gaussian(d_1))

class VanillaPut:
    def __init__(self, stock, expiry, strike):
        self.stock = stock
        self.strike = strike
        self.expiry = expiry

    price = _normal_limits(lambda self, d_1, d_2: \
        -self.stock.spot * math.exp(-self.stock.divid * self.expiry) * \
        norm.cdf(-d_1) + self.strike * math.exp(-self.stock.rate * \
        self.expiry) * norm.cdf(-d_2))

    delta = _normal_limits(lambda self, d_1, d_2: \
        -math.exp(-self.stock.divid * self.expiry) * norm.cdf(-d_1))

    gamma = _normal_limits(lambda self, d_1, d_2: \
        (math.exp(-self.stock.divid * self.expiry) * _gaussian(d_1)) / \
        (self.stock.spot * self.stock.vol * math.sqrt(self.expiry)))

    vega = _normal_limits(lambda self, d_1, d_2: \
        self.stock.spot * math.exp(-self.stock.divid * self.expiry) * \
        math.sqrt(self.expiry) * _gaussian(d_1))

    rho = _normal_limits(lambda self, d_1, d_2: \
        -self.strike * math.exp(-self.stock.rate * self.expiry) * \
        self.expiry * norm.cdf(-d_2))

    theta = _normal_limits(lambda self, d_1, d_2: \
        -self.stock.spot * math.exp(-self.stock.divid * self.expiry) * \
        self.stock.divid * norm.cdf(-d_1) + self.strike * \
        math.exp(-self.stock.rate * self.expiry) * self.stock.rate * \
        norm.cdf(-d_2) - self.stock.spot * math.exp(-self.stock.divid * \
        self.expiry) * (self.stock.vol / (2 * math.sqrt(self.expiry))) * \
        _gaussian(d_1))

class DigitalCall:
    def __init__(self, stock, expiry, strike):
        self.stock = stock
        self.strike = strike
        self.expiry = expiry

    price = _normal_limits(lambda self, d_1, d_2: \
        math.exp(-self.stock.rate * self.expiry) * norm.cdf(d_2))

class DigitalPut:
    def __init__(self, stock, expiry, strike):
        self.stock = stock
        self.strike = strike
        self.expiry = expiry

    price = _normal_limits(lambda self, d_1, d_2: \
        math.exp(-self.stock.rate * self.expiry) * norm.cdf(-d_2))

# add greeks for digital options
