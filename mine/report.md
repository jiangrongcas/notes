# Report

## 20180806

|Date | 20180801 |
|-|-|
|# gul distinct users	| 35344946 |
|# postbuy distinct users	| 26357414 |
|# same distince users | 7146039 |
|# distinct users	| 54556321 |
|# sences in postbuy | 5 (1560, 1640, 2497, 1639, 2113) |
|# postbuy2gul users | 5406324 |
|prob postbuy2gul | 75.65% (5406324 / 7146039) |
|# gul2postbuy users | 4562439 |
|prob gul2postbuy | 63.85% (4562439 / 7146039) |
|

Problem definition：

- When a user triggers the gul:
    - State $s_t$: <log (encoding) of the user (in one day or a finite number of data) $h_t$, trigger information $c_t$>.
    - Action $a_t$ = <information of the called recommend algorithm (match type, weights, recommended product, etc)>.
    - Policy $\mu(s_t)$: may be a set of policies for different scenes.
    - 
    - User's reaction $o_t$: click, close_page, buy, etc.
    - Immediate $r_t$: based on the user's reaction>.
    - Log (encoding) $h_{t+1} = f(h_t, a_t, o_t)$.
    - $Q(s_t, a_t) = r_t + \gamma Q(s_{t+1}, \mu(s_{t+1}))$. 
        - Training tuple <$s_t, a_t, r_t, s_{t+1}$>. If $s_{t+1}$ is predictable, then the training tuple is <$s_t, a_t, r_t$>.
    - 
    - $p(o_t| s_t, a_t)$: predictable
    - $p(c_{t+1}| s_t, a_t)$: partially predictable (e.g. scene id of $c_{t+1}$).
        - $p(s_{t+1}| s_t, a_t)$: unpredictable
  

To do:
- Transition probs
- Correlation bwtween states. Remove uncorrelated state-action pairs in the chain of the Markov chain.
