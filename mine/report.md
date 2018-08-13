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
|prob_postbuy2gul | 75.65% (5406324 / 7146039) |
|# gul2postbuy users | 4562439 |
|prob_gul2postbuy | 63.85% (4562439 / 7146039) |
| 

To do:
- Transition probs
- Correlation bwtween states. Remove uncorrelated state-action pairs in the chain of the Markov chain.

## 20180813

Problems about the probabilities:

|Date | 20180801 |
|-|-|
| # gul distinct users | 35344946 |
| # postbuy distinct users	| 26357414 |
| # same distince users | 7146039 |
| # distinct users	| 54556321 |
| # sences in postbuy | 5 (1560, 1640, 2497, 1639, 2113) |
| prob_gul2postbuy | ``12.91%`` (4562439 / ``35344946``) |
| prob_gul2scene1560 | 0.06% (19996 / 35344946), $\leftarrow$ 20.86% (32256 / 154577) |
| prob_gul2scene1640 | 0.05% (18861 / 35344946), $\leftarrow$ 25.38% (23037 / 90756) |
| prob_gul2scene2497 | 2.42% (854557 / 35344946), $\leftarrow$ 18.12% (1056419 / 5831519) |
| prob_gul2scene1639 | 5.35% (1892044 / 35344946), $\leftarrow$ 19.75% (2077448 / 10518610) |
| prob_gul2scene2113 | 7.38% (2607793 / 35344946), $\leftarrow$ 22.28% (3176376 / 14259414) |
| prob_postbuy2gul | ``20.51%`` (5406324 / ``26357414``) |
| prob_scene1560_to_1640 | 0.22% (333 / 154577), $\leftarrow$ 0.27% (248 / 90756) |
| prob_scene1560_to_2497 | 18.06% (27918 / 154577), $\leftarrow$ 0.12% (7021 / 5831519)|
| prob_scene1560_to_1639 | 11.75% (18168 / 154577), $\leftarrow$ 0.12% (12781 / 10518610) |
| prob_scene1560_to_2113 | 12.72% (19657 / 154577), $\leftarrow$ 0.14% (19746 / 14259414) |
|

- $\sum_{id}$prob_gul2sceneid > prob_gul2postbuy is reasonable, because a user may transfer from gul to different scenes in one day, but the prob_gul2postbuy counts it only once due to the "DISTINCT" command.

Problem definition：
- When a user triggers the gul:
    - State $s_t$: <log (encoding) of the user (in one day or a finite number of data) $h_t$, trigger information $c_t$>.
    - Action $a_t$ = parameters of the recommend algorithm (match type, weights, recommended product, etc).
    - Policy $\mu(s_t) = a_t$: may be a set of policies for different scenes.
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

Note:
- The log $h_t$ should be consistent with the trigger information $c_t$. For example, if the user 
    1) trigger gul_scene_1 with "dress"
    2) trigger gul_scene_2 with "headset" 
    3) trigger gul_scene_1 with "earphone" 
    4) trigger gul_scene_1 with "skirt"

    then there are two Markov chains: 
    1) trigger gul_scene_1 with "dress" $\rightarrow$ trigger gul_scene_1 with "skirt" 
    2) trigger gul_scene_2 with "headset" $\rightarrow$ trigger gul_scene_1 with "earphone".
    
    Is it feasible to quickly screen the consistent log out from the historical data when the gul is triggered?

Key problems:
- How to represent $h_t$? 
    - If we use RNN to encode the history, then how to handle different time intervals between trigger timestamps?
    - time-aware LSTM
    - DNN + attention
- The future reward discount factor should be related to the time interval between current trigger ($t_i$) and next trigger ($t_{i+1}$), e.g., $\gamma^{t_{i+1} - t_i}$, where $t_{i + 1}$ is included in $c_{t_{i + 1}}$, which is unknown.
- How to deal with partial information of the enviroment?
    - Many uncertainties about users, e.g., their mood.
    - [1,2] use RNN to approximate the full state with historical observations and actions.
- How to define $\mu$, value netwok or policy network?
    - Policy gradient methods provide a natural framework for learning policies with continuous, high-dimensional actions [1].
    - RNN for $h$ + policy network [2] or a single recurrent policy network [1].

---
**References**

[1] Wierstra, D., Foerster, A., Peters, J., & Schmidhuber, J. (2007). Solving Deep Memory POMDPs with Recurrent Policy Gradients, 1(1), 697–706. https://doi.org/10.1007/978-3-540-74690-4_71

[2] Hausknecht, M., & Stone, P. (2015). Deep Recurrent Q-Learning for Partially Observable MDPs, 29–37. https://doi.org/10.1.1.696.1421
