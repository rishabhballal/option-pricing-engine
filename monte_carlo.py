import math
import numpy as np

rng = np.random.default_rng()

class StockPaths:
    def __init__(self, count, S_0=100, t=1, r=0.05, d=0.03, v=0.10):
        self.count = count
        self.S_0 = S_0
        self.t = t
        self.r = r
        self.d = d
        self.v = v

    def gbm_solution(self, t=1, rands=[]):
        if not len(rands):
            rands = rng.standard_normal(self.count)
        return [self.S_0*math.exp((self.r - self.d)*self.t - \
            0.5*(self.v**2)*self.t + self.v*math.sqrt(self.t)*x) for x in rands]

    def gbm_steps(self, t=1, N=10**4, rands=[]):
        if not len(rands):
            rands = rng.standard_normal((N,self.count))
        S_t = np.ones((N+1,self.count))*self.S_0
        for i in range(N):
            S_t[i+1] = S_t[i]*(1 + (self.r - self.d)*self.t/N + \
                self.v*math.sqrt(self.t/N) * rands[i])
        return S_t

class OptionValues:
    def __init__(self, S_0, t, r, d, v, payoff):
        self.S_0 = S_0
        self.t = t
        self.r = r
        self.d = d
        self.v = v
        self.payoff = payoff

    # def price(self, count, N=1):
    #     S_t = self.asset_singlestep(count) if N == 1 \
    #         else self.asset_multistep(count, N)
    #     return np.mean([math.exp(-self.r*self.t) * self.payoff(x) for x in S_t])

    def price2(self, count):
        rands = rng.standard_normal(count)
        S = [self.S_0*math.exp((self.r - self.d)*self.t - \
            0.5*(self.v**2)*self.t + self.v*math.sqrt(self.t)*x) \
            for x in rands]
        S_delta = [(self.S_0+0.001)*math.exp((self.r - self.d)*self.t - \
            0.5*(self.v**2)*self.t + self.v*math.sqrt(self.t)*x) \
            for x in rands]
        S_gamma = [(self.S_0+0.001)*math.exp((self.r - self.d)*self.t - \
            0.5*(self.v**2)*self.t + self.v*math.sqrt(self.t)*x) \
            for x in rands]
        S_vega = [(self.S_0+0.001)*math.exp((self.r - self.d)*self.t - \
            0.5*(self.v**2)*self.t + self.v*math.sqrt(self.t)*x) \
            for x in rands]
        S_rho = [(self.S_0+0.001)*math.exp((self.r - self.d)*self.t - \
            0.5*(self.v**2)*self.t + self.v*math.sqrt(self.t)*x) \
            for x in rands]
        S_theta = [(self.S_0+0.001)*math.exp((self.r - self.d)*self.t - \
            0.5*(self.v**2)*self.t + self.v*math.sqrt(self.t)*x) \
            for x in rands]

        self.price = np.mean([math.exp(-self.r*self.t) * self.payoff(S[i]) for i in range(count)])
        self.delta = np.mean([math.exp(-self.r*self.t) * (self.payoff(S_delta[i]) - self.payoff(S[i]))/0.001 for i in range(count)])
        self.gamma = np.mean([math.exp(-self.r*self.t) * (self.payoff(S_delta[i]) - self.payoff(S[i]))/0.001 for i in range(count)])
        self.vega = np.mean([math.exp(-self.r*self.t) * (self.payoff(S_delta[i]) - self.payoff(S[i]))/0.001 for i in range(count)])
        self.rho = np.mean([math.exp(-self.r*self.t) * (self.payoff(S_delta[i]) - self.payoff(S[i]))/0.001 for i in range(count)])
        self.theta = np.mean([math.exp(-self.r*self.t) * (self.payoff(S_delta[i]) - self.payoff(S[i]))/0.001 for i in range(count)])
        return self.price
