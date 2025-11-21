import math
import numpy as np
from scipy.stats import norm

# S - spot, K - strike, t - time to expiry, r - short rate
# d - dividend rate, v - volatility, F - forward price

zc_bond = lambda t,r: math.exp(-r*t)
F = lambda P,t,r,d: P*math.exp((r-d)*t)
forward_contract = lambda S,K,t,r,d : zc_bond(t,r)*(F(S,t,r,d) - K)

def auxiliary_vars(func):
    def wrapper(S,K,t,r,d,v):
        d1 = (math.log(S/K) + (r-d+0.5*v**2)*t)/(v*math.sqrt(t))
        d2 = d1 - v*math.sqrt(t)
        return func(S,K,t,r,d,v,d1,d2)
    return wrapper


# vanilla options

vanilla_call = auxiliary_vars(lambda S,K,t,r,d,v,d1,d2: \
    zc_bond(t,r)*(F(S,t,r,d)*norm.cdf(d1) - K*norm.cdf(d2)))

vanilla_put = auxiliary_vars(lambda S,K,t,r,d,v,d1,d2: \
    zc_bond(t,r)*(-F(S,t,r,d)*norm.cdf(-d1) + K*norm.cdf(-d2)))
    # vanilla_call(S,K,t,r,d,v) - forward_contract(S,K,t,r,d)

digital_call = auxiliary_vars(lambda S,K,t,r,d,v,d1,d2: \
    zc_bond(t,r)*norm.cdf(d2))
    # (vanilla_call(S,K-eps,t,r,d,v) - vanilla_call(S,K+eps,t,r,d,v))/(2*eps)

digital_put = auxiliary_vars(lambda S,K,t,r,d,v,d1,d2: \
    zc_bond(t,r)*norm.cdf(-d2))
    # zc_bond(t,r) - digital_call(S,K,t,r,d,v)


# Greeks of a vanilla call option

n = lambda x: math.exp(-x**2/2)/math.sqrt(2*math.pi)

delta_call = auxiliary_vars(lambda S,K,t,r,d,v,d1,d2: \
    math.exp(-d*t)*norm.cdf(d1))

gamma_call = auxiliary_vars(lambda S,K,t,r,d,v,d1,d2: \
    (math.exp(-d*t)*n(d1))/(S*v*math.sqrt(t)))

vega_call = auxiliary_vars(lambda S,K,t,r,d,v,d1,d2: \
    S*math.exp(-d*t)*n(d1))

rho_call = auxiliary_vars(lambda S,K,t,r,d,v,d1,d2: \
    K*zc_bond(t,r)*t*norm.cdf(d2))

theta_call = auxiliary_vars(lambda S,K,t,r,d,v,d1,d2: \
    S*math.exp(-d*t)*d*norm.cdf(d1) - K*zc_bond(t,r)*r*norm.cdf(d2) - \
    S*math.exp(-d*t)*v*n(d1)/(2*math.sqrt(t)))


# Greeks of a vanilla put option

delta_put = auxiliary_vars(lambda S,K,t,r,d,v,d1,d2: \
    -math.exp(-d*t)*norm.cdf(-d1))
    # delta_call(S,K,t,r,d,v) - math.exp(-d*t)

gamma_put = gamma_call

vega_put = vega_call

rho_put = auxiliary_vars(lambda S,K,t,r,d,v,d1,d2: \
    -K*zc_bond(t,r)*t*norm.cdf(-d2))
    # rho_call(S,K,t,r,d,v) - K*zc_bond(t,r)*t

theta_put = auxiliary_vars(lambda S,K,t,r,d,v,d1,d2: \
    -S*math.exp(-d*t)*d*norm.cdf(-d1) + K*zc_bond(t,r)*r*norm.cdf(-d2) - \
    S*math.exp(-d*t)*v*n(d1)/(2*math.sqrt(t)))
    # theta_call(S,K,t,r,d,v) - (S*math.exp(-d*t)*d - K*zc_bond(t,r)*r)
