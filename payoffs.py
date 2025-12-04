import numpy as np

# usual payoffs for instantiating EuropeanOtion, AmericanOption, BermudanOption
def forward_contract(strike):
    return (lambda S: S - strike)

def vanilla_call(strike):
    return (lambda S: max(S - strike, 0))
def vanilla_put(strike):
    return (lambda S: max(strike - S, 0))

def digital_call(strike):
    return (lambda S: 1 if S > strike else 0)
def digital_put(strike):
    return (lambda S: 1 if S < strike else 0)

def power_call(strike, power):
    return (lambda S: (S - strike)**power if S > strike else 0)
def power_put(strike, power):
    return (lambda S: (strike - S)**power if S < strike else 0)

def straddle(strike):
    return (lambda S: max(S - strike, strike - S))

# usual payoffs for instantiating AsianOption
def arithmetic_asian_call(strike):
    return (lambda S: max(sum(S)/len(S) - strike, 0))
def arithmetic_asian_put(strike):
    return (lambda S: max(strike - sum(S)/len(S), 0))

def geometric_asian_call(strike):
    return (lambda S: max(np.prod(S)**(1/len(S)) - strike, 0))
def geometric_asian_put(strike):
    return (lambda S: max(strike - np.prod(S)**(1/len(S)), 0))

# usual payoffs for instantiating LookbackOption
def lookback_call_fixed(strike):
    return (lambda S, S_min, S_max: max(S_max - strike, 0))
def lookback_put_fixed(strike):
    return (lambda S, S_min, S_max: max(strike - S_min, 0))

def lookback_call_floating():
    return (lambda S, S_min, S_max: S - S_min)
def lookback_put_floating():
    return (lambda S, S_min, S_max: S_max - S)

# usual payoffs for instantiating DiscreteBarrierOption
def discrete_down_and_out_call(strike, barrier):
    return (lambda S: max(S[-1] - strike, 0) if np.all(S > barrier) else 0)
def discrete_down_and_out_put(strike, barrier):
    return (lambda S: max(strike - S[-1], 0) if np.all(S > barrier) else 0)

def discrete_down_and_in_call(strike, barrier):
    return (lambda S: max(S[-1] - strike, 0) if np.any(S < barrier) else 0)
def discrete_down_and_in_put(strike, barrier):
    return (lambda S: max(strike - S[-1], 0) if np.any(S < barrier) else 0)

def discrete_up_and_out_call(strike, barrier):
    return (lambda S: max(S[-1] - strike, 0) if np.all(S < barrier) else 0)
def discrete_up_and_out_put(strike, barrier):
    return (lambda S: max(strike - S[-1], 0) if np.all(S < barrier) else 0)

def discrete_up_and_in_call(strike, barrier):
    return (lambda S: max(S[-1] - strike, 0) if np.any(S > barrier) else 0)
def discrete_up_and_in_put(strike, barrier):
    return (lambda S: max(strike - S[-1], 0) if np.any(S > barrier) else 0)
