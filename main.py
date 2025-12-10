import numpy as np
import stocks
import black_scholes_merton as bsm
import monte_carlo as mc
import trees as tr
import payoffs

# an example of a stock
stock = stocks.Stock(
    spot=100,
    rate=0.05,
    divid=0.00,
    vol=0.25
)

# an example of a vanilla option
option1 = bsm.VanillaPut(
    stock=stock,
    expiry=252,
    strike=95
)
# a similar example but with an early exercise feature
option2 = tr.PathIndependentOption(
    stock=stock,
    expiry=252,
    payoff=payoffs.vanilla_put(strike=95),
    ex_times=range(252)
)
# again, a similar example but with arithmetic averaging over fixed times
option3 = mc.EuropeanOption(
    stock=stock,
    expiry=252,
    payoff=payoffs.arithmetic_asian_put(strike=95),
    path_times=[63, 126, 189, 252]
)

print(f'Prices: {option1.price()}, {option2.price()}, {option3.price()}')


# ---- TO-DO ----
# pandas
# implied vol and volatility surface
# portfolio, replication
# multi-asset options - quantos and margrabes
# interest rate derivatives
