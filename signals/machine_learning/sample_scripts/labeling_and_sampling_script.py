import pandas as pd
import seaborn as sns
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from configs.universe_spec import universe_tech
from data_loading.load_from_disk.load_equities_data import load_equities_data_from_disk
from signals.machine_learning.labelling import get_labels_from_tripple_barrier_returns
from signals.machine_learning.sampling_routines import calculate_sampling_weights_from_avg_label_uniqueness, \
    calculate_average_uniqueness_of_samples, calculate_indicator_matrix, calculate_time_decay
from signals.momentum.time_series_momentum import exp_ma_crossover

# run analysis for a specific stock
sample_eq = 'GOOG'

# load equities data data
eq_data = load_equities_data_from_disk(universe_tech, frequency='B')

prices = eq_data.get_cross_sectional_view('close')[sample_eq].dropna()

# pro-memoria: min_ret means that if a return below min_ret the corresponding feature will never be labeled/sampled
label_info, barrier_dates = get_labels_from_tripple_barrier_returns(prices, span=100, pt_sl=[2, 2],
                                                                    min_ret=0.01, num_threads=8, vb_days=30,
                                                                    meta_label=False)

# add sampling weights from average uniqueness
label_info['w'] = calculate_sampling_weights_from_avg_label_uniqueness(barrier_dates, prices)

# calculate average uniqueness for fun
indicator_matrix = calculate_indicator_matrix(barrier_dates.index, barrier_dates)
avg_sample_uniqueness = calculate_average_uniqueness_of_samples(indicator_matrix)  # how unique is a sample; if 1
time_decay_weights = calculate_time_decay(label_info['w'], maximal_decay=0)  # adjust uniqueness wgts for time decay

# calculate features
x_prices = eq_data.get_cross_sectional_view('close').ffill()

# probably add shape (studying regplot you see that extremes are pretty "extremely" false)
cmas = pd.DataFrame(columns=x_prices.columns, index=x_prices.index)
for feature in cmas:
    cmas[feature] = exp_ma_crossover(x_prices[feature], fast=4, slow=16, window=20, vol_floor=0.)
cmas.dropna(inplace=True)

# training / test split
first_valid_label_index = label_info.first_valid_index()  # first possible y obs
first_valid_cma_index = cmas.first_valid_index()  # first possible X (i.e. feature obs)
if first_valid_cma_index > first_valid_label_index:
    label_info = label_info.loc[first_valid_cma_index:]  # ensure (X,y) pair corresponds

X_train, X_test, y_train, y_test = train_test_split(
    cmas.loc[label_info.index], label_info['bin'], shuffle=False, test_size=0.2, random_state=None)

# -------------------------------------------
# Random Forest fit (3 options)
# -------------------------------------------

clf1 = DecisionTreeClassifier(criterion='entropy', max_features='auto', class_weight='balanced')  # the simple learner
clf1 = BaggingClassifier(base_estimator=clf1, n_estimators=100, max_features=1.,
                         max_samples=avg_sample_uniqueness.mean(), oob_score=True)  # combine the simple learners

clf2 = RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample', criterion='entropy')

clf3 = RandomForestClassifier(n_estimators=1, criterion='entropy', bootstrap=False, class_weight='balanced_subsample')
clf3 = BaggingClassifier(base_estimator=clf3, n_estimators=100, max_samples=avg_sample_uniqueness.mean(),
                         max_features=1.)

# fit (pass the sampling probabilities calculated above, e.g. time_decay_weights)
clf1.fit(X=X_train, y=y_train, sample_weight=time_decay_weights.loc[y_train.index])
clf2.fit(X=X_train, y=y_train, sample_weight=time_decay_weights.loc[y_train.index])
clf3.fit(X=X_train, y=y_train, sample_weight=time_decay_weights.loc[y_train.index])

# score / evals
print(clf1.score(X=X_train, y=y_train))  # avg log-likelihood
predictions = clf3.predict(X_test)
probabilities = clf3.predict_proba(X_test)
print(f1_score(y_test, predictions))
print(recall_score(y_test, predictions))
print(precision_score(y_test, predictions))

# example plot
# regress probability of label on tb-return
pos_prob = pd.Series(probabilities[:, 0], index=y_test.index)
df = pd.concat([pos_prob, y_test], axis=1)
df.columns = ['pp', 'lbl']
df['ret'] = label_info['return'].loc[df.index]
sns.regplot(x=df.pp, y=df.ret)


# clf2.score(X=X_test.to_frame(), y=y_test)

# graphing
# from sklearn.externals.six import StringIO
# from IPython.display import Image
# from sklearn.tree import export_graphviz
# import pydotplus
# dot_data = StringIO()
# export_graphviz(clf1.estimators_[0], out_file='C:/Users/28ide/tree.dot',
#                   filled=True, rounded=True,
#                   special_characters=True)
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# Image(graph.create_png())

# Convert to png
# from subprocess import call
# # call(['dot', '-Tpng', 'C:/Users/28ide/tree.dot', '-o', 'tree.png', '-Gdpi=600'])
# #
# # # Display in python
# # import matplotlib.pyplot as plt
# # plt.figure(figsize = (14, 18))
# # plt.imshow(plt.imread('tree.png'))
# # plt.axis('off');
# # plt.show();
# pca decomp, centered but not scaled (i.e. demeaned)
