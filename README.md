# Kaggle Caterpillar Tube Pricing Challenge

This repo contains my solution to the [Caterpillar Tube
Pricing](https://www.kaggle.com/c/caterpillar-tube-pricing) Kaggle competition,
as well as all the exploration code that I produced along the way. I finished
with a score of 0.211679 on the private leaderboard, ranking #31 out of 1323
participating teams (top 3%), or #13 among individual participants.

The main solution code is in `soln/`, and the exploratory IPython notebooks are
in `exploration/`. My final model was a bagged ensemble of xgboost tree models,
as detailed under "Bagging experiment" below. The rest of this file describes
my approach to the competition, and other ideas that I've tried but that did
not make it into the final model. I am assuming familiarity with the
[competition setup](https://www.kaggle.com/c/caterpillar-tube-pricing), [error
metric](https://www.kaggle.com/c/caterpillar-tube-pricing/details/evaluation),
and [data](https://www.kaggle.com/c/caterpillar-tube-pricing/data).


## Cross-validation approach

I split the training set into K=10 folds for cross validation.

Because of bracket pricing, the training set can have multiple rows for the
same `tube_assembly_id`. However, the `tube_assembly_id`s in the training set
and the test set are disjoint. To preserve this in my K-fold split, I split the
unique `tube_assembly_id`s into K folds, instead of splitting the rows into K
folds.

I also noticed that the histogram of the `supplier` column in the test set
closely matched the histogram of the same column in the training set. To
preserve this in my K-fold split, I stratified by supplier. The code for all
this can be found in the `add_dev_fold_column` function in `soln/dataset.py`.

It paid off to do the split very carefully this way, because my
cross-validation procedure gave me a good estimate of generalization error. My
mean cross-validation RMSLE was 0.213 (stdev 0.017), which was very close to
the final score of 0.212. In contrast, my public leaderboard score was 0.219,
which was less informative. People who optimized for the public leaderboard
score might have been overfitting, because my solution jumped 19 spots higher
on the private leaderboard (#50 on public vs. #31 on private).


## Feature engineering

The dataset came as a set of relational tables, with interesting opportunities
for feature engineering. I picked all the obvious columns from `train_set.csv`
and `tube.csv`, and used one-hot encoding for the categorical features. (I
dropped any feature that had a support of less than `min_seen_count=10` rows.)

I noticed that the `min_order_quantity` and `quantity` columns were
inconsistent among different suppliers, so I defined an `adjusted_quantity`
feature that took the max of those columns. I also added a `bracketing_pattern`
categorical feature that captured the set of quantities for a
`tube_assembly_id` that occurred with bracket pricing, e.g. `(1, 2, 5, 10, 25,
50, 100, 250)` for `TA-00222`.

In addition, I extracted list-of-categoricals features for the specs in
`specs.csv` and the components in `bill_of_materials.csv`. For example,
`TA-00015` had specs `['SP-0063', 'SP-0069', 'SP-0080']` and components
`['C-0448', 'C-0448', 'C-0449', 'C-0449']` (duplicates to account for the count
of each component). I converted these list-of-categorical features to integer
features that count how many time each spec or component appeared in a tube
assembly, and discarded any features seen less than `min_seen_count` times.

In addition to using the `component_id`s in the `components` feature, I added a
`component_types` feature that stored the "type" of each component (as given by
the `component_type_id` column, e.g. `CP-028`), and a `component_groups`
feature that stored the "group" for each component (as given the csv file it
came from, e.g. `comp_adaptor.csv`). This means that I had three layers of
granularity for the component features: each `component_id` belonged to a
`component_type`, and each `component_type` belonged to a `component_group`.

Not all component groups had the same columns, but using sensible defaults
(e.g. False for `unique_feature`, zero for `weight`, etc), I extracted the
following additional features for each `tube_assembly_id`:

- `unique_feature_count`: number of components with `unique_feature=True`
- `orientation_count`: number of components with `orientation=True`
- `groove_count`: number of components with `groove=True`
- `total_component_weight`: sum of weights of all components
- `component_end_forms`: list-of-categoricals combining columns like
  `end_form_id_1`, `end_form_id_2`, etc.
- `component_connection_types`: list-of-categoricals combining columns like
  `end_form_id_1`, `end_form_id_2`, etc.
- `component_part_names`: list-of-categoricals combining the strings from the
  `part_name` column in `comp_other.csv`
- `component_max_length`
- `component_max_overall_length`
- `component_max_bolt_pattern_wide`
- `component_max_bolt_pattern_long`
- `component_max_thickness`
- `component_min_thread_pitch`
- `component_min_thread_size`

Finally, I added a list-of-categoricals `ends` feature combining `end_a` and
`end_x`, with the intuition being that the end forms might be interchangeable,
and so their ordering might not matter. I also added a few features computed
using simple physics:

- `physical_volume`: `length * pi * ((diameter / 2) ^ 2)`
- `inner_radius`: `diameter / 2 - wall_thickness`
- `material_volume`: `physical_volume - length * pi * (inner_radius ^ 2)`

The code for all this can be found in `soln/dataset.py`. My approach to testing
was to sanity-check each feature as I added it, using the notebook
`exploration/check_featurizer.ipynb`.


## Choosing a model and optimizing hyperparameters

(A lower RMSLE is better. All RMSLE values here are obtained via cross
validation.)

I started out by trying `RandomForestRegressor` and `ExtraTreesRegressor` from
scikit-learn. I got the best results (about 0.240 RMSLE) with 100 trees of
unrestricted depth, and selecting 40% of available features for each tree.

I then switched to `xgboost`, which immediately brought my RMSLE to 0.225 with
1000 trees. There were diminishing returns from adding more trees, with the
best RMSLE of about 0.213 with 10K trees. My other parameters were: `gamma =
0`, `eta = 0.02`, `max_depth = 8`, `min_child_weight = 6`, `subsample = 0.7`,
`colsample_bytree = 0.6`. I got some of these ballpark values by reading the
Kaggle forums.

I also tried searching the parameter space for parameters that would lower my
RMSLE. I fired up an 8-core `c4.2xlarge` instance on EC2, which was much faster
than my dual-core laptop, since `xgboost` is multithreaded. (With spot pricing,
I paid less than 10 cents per hour, whereas the current regular price is 44
cents per hour.) I used `hyperopt` to guide the search, defining appropriate
ranges for the parameters I thought would be most salient (see
`soln/optimize_params.py` for an example). The search did not reveal any large
improvement in RMSLE, confirming that my parameter were already set pretty
well.

My workflow for searching parameters involved dumping the feature matrices for
the K folds to disk (`soln/dump_xv_folds.py`), so that the script performing
the search (`soln/optimize_params.py`) did not have to repeat the work of
featurizing.


## Bracketing experiment

In this experiment I tried to take advantage of the bracket-pricing structure
present in the data. I noticed that a majority of the rows (58% of the training
set and 58% of the test set) came from tubes with the bracketing pattern `(1,
2, 5, 10, 25, 50, 100, 250)`, which I will call the well-behaved bracket. All
of these tubes came from supplier `S-0066`. For all of the tubes in the
well-behaved bracket, the prices can be explained by a simple linear model:

    total_cost(tube) = fixed_cost(tube) + variable_cost(tube) * quantity(tube)

where the `cost` column in the data is the per-unit cost, i.e.
`total_cost(tube) / quantity(tube)`. This linear relation holds with an `r2 >
0.9999` for all the tubes in the well-behaved bracket. For example, for tube
`TA-00002`:

![bracket pricing figure](/images/bracket.png?raw=true)

(A similar linear relation holds for a few other bracket patterns, but not all
of them.)

Of course, the `fixed_cost` and `variable_cost` are unobserved, but they can be
recovered from the training set by fitting a linear model, as illustrated
above. What is more interesting is that the fixed costs for all the tubes in
the well-behaved bracket cluster cleanly into four clusters:

![fixed costs figure](/images/fixed_costs.png?raw=true)

This suggests a two-step approach: First, classify a tube into one of the four
fixed-cost clusters. Second, predict the variable cost for the tube. Finally,
combine these two predictions to get the final cost for any quantity. The code
for this is in `exploration/bracket_pricing_*.ipynb`.

I built the fixed cost classifier using `xgboost` with a softmax objective
function (four classes), and achieved an accuracy of about 94%. For the
variable cost regressor, I again used `xgboost`, choosing to optimize the MSE
of `log(variable_cost + 1)`. This objective function is an approximation, since
when we predict the variable cost separately, we are no longer directly
optimizing RMSLE on the original data. (Intuitively, the same amount of error
in `variable_cost` will contribute a different amount of error to the final
RMSLE, depending on the tube's `fixed_cost`.)

Using this combined model, I achieved a small improvement in RMSLE for the
well-behaved bracket, compared to my original `scikit-learn` model (which did
not model bracketing separately). However, the improvement vanished when I
compared against a tuned `xgboost` model with 10K trees. Thus, despite all this
interesting structure in the data, simple boosted trees still did better than
my combined modeling approach.


## Component clustering experiment

In this experiment I tried to tie together the features from similar
components. Of the 2047 known components, many occur in only one or two tubes.
Even more alarmingly, hundreds of components occur in test set tubes but not in
train set tubes, and vice versa:

<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>test_seen_count</th>
      <th>0..0</th>
      <th>1..1</th>
      <th>2..5</th>
      <th>5..10</th>
      <th>10..20</th>
      <th>20..50</th>
      <th>50..100</th>
      <th>100..inf</th>
    </tr>
    <tr>
      <th>train_seen_count</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0..0</th>
      <td>346</td>
      <td>399</td>
      <td>81</td>
      <td>2</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>1..1</th>
      <td>407</td>
      <td>140</td>
      <td>83</td>
      <td>9</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>2..5</th>
      <td>111</td>
      <td>94</td>
      <td>112</td>
      <td>38</td>
      <td>2</td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>5..10</th>
      <td>2</td>
      <td>5</td>
      <td>43</td>
      <td>37</td>
      <td>8</td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>10..20</th>
      <td></td>
      <td></td>
      <td></td>
      <td>13</td>
      <td>41</td>
      <td>2</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>20..50</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>9</td>
      <td>23</td>
      <td>1</td>
      <td></td>
    </tr>
    <tr>
      <th>50..100</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>3</td>
      <td>6</td>
      <td></td>
    </tr>
    <tr>
      <th>100..inf</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>30</td>
    </tr>
  </tbody>
</table>

I reasoned that tying similar components together could alleviate this sparsity
problem. I focused my investigation on components in the `comp_straight.csv`
group, which was one of the worst offenders with regards to sparsity. The code
for this is in `exploration/components.ipynb` and
`exploration/straight_cluster.ipynb`. I tried three different ways to tie
components together:

- K-means clustering;
- Agglomerative clustering, followed by extracting flat clusters;
- Explicitly mapping uncommon `component_id`s to their nearest neighbor that is
  sufficiently common.

For each of these, I tried using the original component features + the tied
component features, or only the tied component features. Despite reducing
sparsity, these techniques did not give me a significant improvement in RMSLE.


## Expert ensemble experiment

Early on in the competition, I noticed that I could get slightly better RMSLE
by training separate models on subsets of the training set. For example, if I
trained a model on all tubes from supplier 72, it would be better at predicting
prices from the same supplier, than the model that was trained on data from all
suppliers. I call such a model an "expert", since it specializes on a specific
subset of the training set.

As a first experiment, I trained a base model on everything, and an expert
model on all tubes with uncommon brackets (i.e., all tubes with a
`bracketing_pattern` other than `(1, 2, 5, 10, 25, 50, 100, 250)`, `(1, 6,
20)`, `(1, 2, 3, 5, 10, 20)`, `(1, 2, 5, 10, 25, 50, 100)`, or `(5, 19, 20)`).
At prediction time I used the base model or the expert model, depending on
whether the instance had a common or uncommon `bracketing_pattern`. This gave
me an improvement of about 0.0006 in RMSLE, both in my cross validation and on
the public leaderboard.

As a second experiment, I trained a base model on everything, and a separate
expert model for each of the five most common suppliers (66, 41, 72, 54, 26).
For each of these expert models, I used cross-validation to determine whether
the expert model did better than the base model. Including only the experts
that beat the base model, I ended up with experts for suppliers 41, 72, and 54.
This gave me a small boost in cross-validation RMSLE, but the variance was
high, and my public score was worse than the base model. (As it turns out, my
private score improved, but there was no way to know this until the competition
was over. In any case, the uncommon-brackets expert gave an improvement that
was more robust than the supplier-specific experts.)


## Bagging experiment

As a final experiment, I checked whether I could improve my score using
bagging. After trying a few different options, I settled on training models on
9 bags, where I obtained each bag by sampling 90% of `tube_assembly_id`s in the
training set, without replacement. (This is not the same as taking 90% of rows;
see the discussion about cross-validation earlier.) I then averaged the
predictions given by the 9 models to get the final prediction. (Taking the
median worked slightly worse.) This simple technique gave me a boost of 0.0008
in RMSLE over the base model with 10K trees.
