# Kaggle Caterpillar Tube Pricing Challenge

This repo contains my solution to the [Caterpillar Tube
Pricing](https://www.kaggle.com/c/caterpillar-tube-pricing) Kaggle challenge,
as well as all the exploration code that I produced along the way. I finished
with a score of 0.211679 on the private leaderboard, ranking #31 out of 1323
participating teams (top 3%).

The main solution code is in `soln/`, and the exploratory IPython notebooks are
in `exploration/`. My final model was a bagged ensemble of xgboost tree models,
as detailed under "Bagging experiment" below. The rest of this file describes
other ideas that I have tried for this challenge.


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


## Bracketing experiment


## Component clustering experiment


## Expert-ensemble experiment


## Bagging experiment


## Hyperopt workflow
