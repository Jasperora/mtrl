
# MTRL
Multi Task RL Algorithms

the repo is copied from: https://github.com/facebookresearch/mtrl

## Update
1. added wandb logger, please check https://wandb.ai/cwz19/mtrl/overview?workspace=user-cwz19 for more history experiments results.
2. added video logger, the env will save video of the result every n step. 
3. changed img input to vector input
4. used SAC instead of SAC_ae, when training distral. (you may use SAC_ae by changing the config)

# some problems

1. 現在seed沒有辦法固定，也就是用同一個seed跑實驗會得到不同的曲線(實驗無法付現)
2. eval的時候效果不好，現在生成影片用的是train的時候的結果，eval的代碼可能存在問題，還沒解決
3. backward的效果顯著低於forward (其他動作沒試，估計效果也部會太好)
4. 同時訓練兩個動作時，效果不好
