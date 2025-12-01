# some standard option payoffs

forward_contract = lambda K: (lambda S: S-K)

vanilla_call = lambda K: (lambda S: S-K if S>K else 0)
vanilla_put = lambda K: (lambda S: K-S if S<K else 0)

digital_call = lambda K: (lambda S: 1 if S>K else 0)
digital_put = lambda K: (lambda S: 1 if S<K else 0)

power_call = lambda K, l: (lambda S: (S-K)^l if S>K else 0)
power_put = lambda K, l: (lambda S: (K-S)^l if S<K else 0)

straddle = lambda K: (lambda S: S-K if S>K else K-S)

asian_call = lambda K: (lambda S: sum(S)/len(S) - K if sum(S)/len(S) > K else 0)
asian_put = lambda K: (lambda S: K - sum(S)/len(S) if sum(S)/len(S) < K else 0)
