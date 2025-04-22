preprossesed MINDsmall_train dataset

chose user features of ['history_len', 'hour', 'day_of_week', 'is_weekend']

chose context features of clustered category and subcategory OHE
did this with sentence transformer and KMeans clusters

Initially trained a logistic regression for predicting click/no-click for articles, then a DQN for slates

Trained a contextual linucb bandit and a contextual thompson sampling bandit

added batch updating

added weight visualizations

