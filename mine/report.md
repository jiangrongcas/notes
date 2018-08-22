TOC

[toc]
# Multi-Scene Recommendation Optimization
## 20180801
- click_time（操作时间）：不一定有点击发生？
- 无线价格的无线是什么意思？
- promotion_price（当日促销价）是什么意思？
- 页面类型和场景id有什么不同？比如页面类型的Page_Home不算作一个场景？
- 如何知道用户是从哪个场景进入当前场景的？（比如计算从场景A转换到场景B的概率需要知道这些信息）

| 序号 | gul_track_log_d_ext | postbuy_quarter_impression_log_skip_above_beili | 区别 |
|-|-|-|-|
| 1 | 应用ID（app_id）| | N/A |
| 2 | 算法曝光接口ID（pvid）| pvid（pvid） | 无 |
| 3 | 用户ID（user_id）| 用户ID（user_id）| 无 |
| 4 | 算法接口类型（match_type）| match类型（matchtype） | |
| 5 | 展示位置（position）| 位置（position）| |
| 6 | 页面类型（page）| 场景id（sceneid）| |
| 7 | 事件类型（event_id）| 触发item（trigger_id）| |
| 8 | 参数1（arg1）| arg1（arg1）| 无 |
| 9 | UT日志埋点参数（args）| args（args）| 无 |
| 10 | 商品ID（auction_id）| 宝贝id（auction_id）| 无 |
| 11 | 操作时间（click_time）| 本地时间（local_time）| |
| 12 | 无线价格（price）| 当日促销价（promotion_price）| |
| 13 | 预估ctr（ctr）| ctr（ctr）| |
| 14 | 预估cvr（cvr）| cvr（cvr）| |
| 15 | 日期（ds）| ds（ds）| 无 |
| 16 | | 订单商品id（context_id）| N/A |
| 17 | | rankscore（score）| N/A |

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

Multiagent RL: 
- The two agents have different objects and each one aims to maximize its own rewards considering the reaction of the other. 
- There is not a global object.
-  [1] designs an algorithms to find the Nash equilibrium for the two agents. Guarantee to converge.

Proposal
- If we view *Platform* and *User* as two agents with different objectives, will the framework applicable? 
    - Hard to define the rewards of users.
- Are the probabilities of scene transformations useful in the algorithm design? 
    - First, use existing data to train a scene-transition-prob network with SL; Second, use RL to refine the probs. This network can be used to build a model-based learning framework.
- match_type: hierarchical (multi-task) RL: 1) which algorithm type + 2) algorithm  parameters. Further, what's the difference between algorithms in different scenes and with different match types? Just different network architectures like [2]?
    - For example, cellphone -> match, dress -> similar
- How to cooperate? Setting a global Q-value seems too simple, is it efficient?
  
Next:
- Transition probs
- Think about correlation bwtween states.

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

- $\sum_{id}$prob_gul2sceneid > prob_gul2postbuy is reasonable, because a user may transfer from gul to different scenes in one day, but the prob_gul2postbuy counts it only once due to the "DISTINCT" command.

Model:
- When a user triggers the gul:
    - Available information for taking actions: 
        - State $s_t$ = $g$(historical log $h_t$, trigger information $c_t$), where $g(\cdot)$ is a function using available (partial) information to approximate the (complete) state.
        - Action $a_t$ = parameters of the recommend algorithm (match type, weights, recommended product, etc).
        - Policy $\mu(s_t) = a_t$: may be a set of policies for different scenes.
    - Information after taking actions: 
        - User's reaction $o_t$: click, close_page, buy, etc.
        - Immediate $r_t$: based on $o_t$.
        - Updating history $h_{t+1} = f(h_t, a_t, o_t, c_t)$, where $f(\cdot)$ is a function (RNN, LSTM, simple concatenation, etc).
     - $Q(s_t, a_t) = r_t + \gamma Q(s_{t+1}, \mu(s_{t+1}))$. 
        - Model-free training tuple: <$s_t, a_t, r_t, s_{t+1}$>. 
        - If both $p(o_t| s_t, a_t)$ and $p(c_{t+1}| s_t, a_t)$ are available, then $s_{t+1}$ = <$h_{t+1}$, $c_{t+1}$> is predictable, leading to a model-based training tuple: <$s_t, a_t$>.

Note:
- The log $h_t$ should be consistent with the trigger information $c_t$. For example, if the user 
    1) trigger gul_scene_1 with "dress",
    2) trigger gul_scene_2 with "headset",
    3) trigger gul_scene_1 with "earphone",
    4) trigger gul_scene_1 with "skirt",

    then there are two Markov chains: 
    1) trigger gul_scene_1 with "dress" $\rightarrow$ trigger gul_scene_1 with "skirt",
    2) trigger gul_scene_2 with "headset" $\rightarrow$ trigger gul_scene_1 with "earphone".

Key problems:
1) How to represent $h_t$, i.e., how to define $f(\cdot)$? 
    - If we use RNN to encode the history, then how to handle different time intervals between trigger timestamps?
        - Possible solutions: time-aware LSTM [3], DNN + attention
2) The future reward discount factor should be related to the time interval between current trigger ($t_i$) and next trigger ($t_{i+1}$), e.g., $\gamma^{t_{i+1} - t_i}$, where $t_{i + 1}$ is a componnet of $c_{t_{i + 1}}$ and is unknown.
    - Possible solutions: predict next trigger time [4].
3) How to deal with partial information of the enviroment (uncertainties about users, e.g., their mood), e.g., how to define $g(\cdot)$?
    - Possible solutions: Using RNN to approximate the full state with historical observations and actions [1, 2].
4) How to define $\mu$, value netwok or policy network?
    - Policy gradient methods provide a natural framework for learning policies with continuous, high-dimensional actions [1].
    - RNN for $h$ + policy network [2] or a single recurrent policy network [1].

Next:

- Read [3, 4] and related references.
- Give a detailed document, presenting related works and highlighting our novelties/contributions/methods. 

---
**References**

[1] Wierstra, D., Foerster, A., Peters, J., & Schmidhuber, J. (2007). Solving Deep Memory POMDPs with Recurrent Policy Gradients, 1(1), 697–706. https://doi.org/10.1007/978-3-540-74690-4_71

[2] Hausknecht, M., & Stone, P. (2015). Deep Recurrent Q-Learning for Partially Observable MDPs, 29–37. https://doi.org/10.1.1.696.1421

[3] Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction. Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.

[4] Wang J, Zhang Y. Opportunity model for e-commerce recommendation: right product; right time. Proceedings of the 36th international ACM SIGIR conference on Research and development in information retrieval. ACM, 2013: 303-312.

## 20180820

Movtivation:
- There are usually many scenes in an e-commerce platform and users can switch between the scenes, while existing recommender systems target at maximizing the immediate (or one-shot) profit at a single scene and lack efficient representation of historical data. Thus, their solutions are suboptimal for the platform to gain maximal overall (long-term) reward.

Key problems:
1) How to represent $h_t$, i.e., how to define $f(\cdot)$? 
    - Possible solutions: weighted sum pooling LSTM [3]
        - [3] uses the DIN (Deep Interest Network) to performs as a weighted sum pooling of visited cates (historical user behavior), but did not consider historical ads and the time stamp of each visit.
    - RNN to encode the history. Then how to handle different time intervals between trigger timestamps?
        - Design an efficient RNN-based structure, which is a challenging and interesting task.
    - DNN + attention

2) The future reward discount factor should be related to the time interval between current trigger ($t_i$) and next trigger ($t_{i+1}$), e.g., $\gamma^{t_{i+1} - t_i}$, where $t_{i + 1}$ is a componnet of $c_{t_{i + 1}}$ and is unknown.
    - Possible solutions: predict next trigger time [4].
        - [4] borrows some models from statistics (hazards model, survival analysis, variational Bayesian, etc) to calculate the probability p(when to buy A | A will be bought).
    - Something wrong with this formula. My preliminary idea is that users' impression is time-interval-related, which can be captured by the representation of $h$, but this formula emphasizes the reward. In the real world, $t_{i+1} - t_i$ would not be very large, so it seems useless to introduce such a complex formula.

3) How to define $\mu$, value netwok or policy network?
    - If the action space $A$ consists of all products, then policy network is infeasible because the ouput layer need to contains $|A|$ neurons.
    - If the actions are weights of several LTR features, then policy network works.
        - LTR features of good $e$ at state $s$: $\pi(s, e)$.
        - Weights output by the DDPG policy at state $s$ [1]: $\mu(s)$.
        - Score of good $e$ at state $s$: $\mu(s)\pi(s, e)$.
        - Since the ranking list is based on both $\mu(\cdot)$ and $\pi(\cdot)$, the immediate reward $r$ is related to both $\mu(\cdot)$ and $\pi(\cdot)$. Because $\mu(\cdot)$ aims to maximize the discount overall reward, it relies on on $\pi(\cdot)$. 
        - If $\pi(\cdot)$ is a bad algorithm (e.g., totally random), then $\mu(\cdot)$ would not be good.
    - Actions can also be goods. Value network. Q-learning.
        - $\mu(s) = \arg\max_a Q(s, a)$, where $Q(s, a)$ is a value network.
        - We prefer this method because it does not depent on the LTR's features.

4) Model-free or model-based?
    - [2] changes the model-free framwork to a model-based one through estimating the state transition probabilities.
        - It mentions that "maintaining an approximate model of the environment to help with performing reliable updates of value functions."
        - Since it considers only one search session (i.e., a session from a query's being input to the user's leaving), its state representation is simple, which consists of historical user reactions (conversion, abandon, continuing) to displayed items.
        - To change to model-baseed framework, it just needs to estimate the probabilities of user reactions to displayed items (i.e., $p(o_t| s_t, a_t)$).
    - To make our framework model based, we need to know $p(c_t| s_t, a_t)$, which is hard to estimate. So we prefer to develop model-free algorithmes at present.

Next:
We will focus on problem 1. The 2nd possible solution (RNN) is challenging, which may produce a big news if we can address it. The 3rd possible solution (DNN) is relatively easy to implement. We will try the two proposals simultaneously.

---
**References**
[1] Feng, J., Li, H., Huang, M., Liu, S., Ou, W., Wang, Z., & Zhu, X. (2018). Learning to Collaborate : Multi-Scenario Ranking via Multi-Agent Reinforcement Learning *. Www, 1939–1948.

[2] Melo, A., & Paulheim, H. (2017). Reinforcement Learning to Rank in E-Commerce Search Engine: Formalization, Analysis, and Application, 2065(July 2017), 54–57. https://doi.org/10.1145/nnnnnnn.nnnnnnn

[3] Zhou, G., Song, C., Zhu, X., Fan, Y., Zhu, H., Ma, X., … Gai, K. (2017). Deep Interest Network for Click-Through Rate Prediction. Retrieved from http://arxiv.org/abs/1706.06978

[4] Wang, J., & Zhang, Y. (2013). Opportunity model for e-commerce recommendation: Right Product, Right Time. Proceedings of the 36th International ACM SIGIR Conference on Research and Development in Information Retrieval - SIGIR ’13, 303. https://doi.org/10.1145/2484028.2484067