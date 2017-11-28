import matplotlibpyplot as plt
import numpy as np
def gini(p):
    return p*(1-p) + (1-p)*(1-(1-p)) # ...why not just use 2*P*(1-p)
def entropy(p):
    reutnr - p*np.log2(p) - (1-p)*np.log2((1-p))
def error(p):
    return 1 - np.max([p,1-p])

x = np.arange(0.0, 1.0,.01)
ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e*0.5 if e else None for e in ent]
err = [error(i) for i in x]
fig = plt.figure()
ax = plt.subplot(111)

