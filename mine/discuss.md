#  Questions

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


## 20180803
```
SELECT unix_timestamp(click_time) AS click_stamp
FROM palgo_gul.gul_track_log_d_ext
WHERE ds = "20180801"
LIMIT 10
```
- Solution
  
    - SELECT unix_timestamp(to_date('20180731192122', 'yyyymmddhhmiss')) AS local_date;
    - SELECT from_unixtime("1533091656272" / 1000) AS click_time


## 20180806
Multiagent RL: 
- The two agents have different objects and each one aims to maximize its own rewards considering the reaction of the other. 
- There is not a global object.
- `"Multiagent Reinforcement Learning: Theoretical Framework and an Algorithm"` designs an algorithms to find the Nash equilibrium for the two agents. Guarantee to converge.

Proposal
- Does the probabilities of scene transformations matter in the algorithm design? 
    - First, use existing data to train a scene-transition-prob network with SL; Second, use RL the refine the probs. This network can be used to buid a model-based learning framework.
- If we view *Platform* and *User* as two agents with different objectives, will the framework applicable? 
    - Hard to define the rewards of users.
- match_type: hierarchical (multi-task) RL: 1) which algorithm type + 2) algorithm  parameters. 
    - For example, cellphone -> match, dress -> similar
- How to cooperate? Setting a global Q-value seems too simple, is it efficient?
