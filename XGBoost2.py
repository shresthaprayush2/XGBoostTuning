import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dtrain = xgb.DMatrix(X_train, label=y_train)


#HyperParameter tuning with Randomized Search
# from sklearn.model_selection import RandomizedSearchCV
#
# # Define the grid of parameters to search
# param_grid = {
#     'n_estimators': [50, 100, 150],
#     'max_depth': [3, 5, 7],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'subsample': [0.8, 1.0],
#     'colsample_bytree': [0.8, 1.0]
# }
#
# random_search = RandomizedSearchCV(
#     estimator=xgb_model,
#     param_distributions=param_grid,
#     scoring='accuracy',
#     cv=3,
#     n_iter=20,  # Number of random combinations to try
#     verbose=1,
#     n_jobs=-1
# )
# startTime = time.time_ns()
# random_search.fit(X_train, y_train)
# print(f"Best Parameters: {random_search.best_params_}")
#
# endTime = time.time_ns()
#
# print(f'The Time Taken Using Grid Search Is {endTime-startTime}')

# Convert to DMatrix (XGBoost's optimized data structure)
# Parameter grid

param_grid = {
    'max_depth': [3, 5, 7],
    'eta': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

num_boost_round = 100
early_stopping_rounds = 10

# Store results
results = []
best_params = {}
best_score = float('inf')

# Loop through parameter grid
for max_depth in param_grid['max_depth']:
    for eta in param_grid['eta']:
        for subsample in param_grid['subsample']:
            for colsample_bytree in param_grid['colsample_bytree']:
                params = {
                    'objective': 'multi:softmax',
                    'num_class': 3,
                    'max_depth': max_depth,
                    'eta': eta,
                    'subsample': subsample,
                    'colsample_bytree': colsample_bytree,
                    'eval_metric': 'mlogloss',
                    'seed': 42
                }

                # Cross-validation
                cv_results = xgb.cv(
                    params=params,
                    dtrain=dtrain,
                    num_boost_round=num_boost_round,
                    nfold=3,
                    metrics='mlogloss',
                    early_stopping_rounds=early_stopping_rounds,
                    as_pandas=True,
                    seed=42
                )

                # Record results
                mean_test_score = cv_results['test-mlogloss-mean'].min()
                best_iteration = cv_results['test-mlogloss-mean'].idxmin()
                print(f"Params: {params}, LogLoss: {mean_test_score:.4f}, Best Iteration: {best_iteration}")

                # Adding the result in the dictionary
                results.append({
                    'max_depth': max_depth,
                    'eta': eta,
                    'subsample': subsample,
                    'colsample_bytree': colsample_bytree,
                    'logloss': mean_test_score
                })
                if mean_test_score < best_score:
                    best_score = mean_test_score
                    best_params = params

# Convert results to DataFrame for easy handling and visualization
results_df = pd.DataFrame(results)
print("\nBest Parameters:")
print(best_params)
print(f"Best CV LogLoss: {best_score:.4f}")

print(results_df)
# Plotting function
def plot_parameter_impact(parameter, title):
    plt.figure(figsize=(8, 6))
    for value in sorted(results_df[parameter].unique()):
        subset = results_df[results_df[parameter] == value]
        plt.plot(subset.index, subset['logloss'], label=f'{parameter}={value}')
    plt.xlabel(f"Parameter Combination Index {title}")
    plt.ylabel("Log-Loss")
    plt.title(f"Impact of {title} on Log-Loss")
    plt.legend()
    plt.savefig(f'{title}.png')


# Plot individual parameter impact
plot_parameter_impact('max_depth', "Max Depth")
plot_parameter_impact('eta', "Learning Rate (Eta)")
plot_parameter_impact('subsample', "Subsample Ratio")
plot_parameter_impact('colsample_bytree', "Column Subsample")
