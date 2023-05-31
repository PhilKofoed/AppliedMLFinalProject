from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
def DecisionTree_CrossValidation_lightgbm(num_leaves, max_depth, learning_rate, n_estimators, data, targets):
    """Decision Tree cross validation.
       Fits a Decision Tree with the given paramaters to the target 
       given data, calculated a CV accuracy score and returns the mean.
       The goal is to find combinations of max_depth, min_samples_leaf 
       that maximize the accuracy
    """
    
    estimator = LGBMClassifier(random_state=42, 
                                       num_leaves=num_leaves, 
                                       max_depth=max_depth, 
                                       learning_rate=learning_rate, 
                                       n_estimators=n_estimators)
    
    cval = cross_val_score(estimator, data, targets, scoring='accuracy', cv=5)
    
    return cval.mean()
def optimize_DecisionTree_lightgbm(data, targets, pars, n_iter=5):
    """Apply Bayesian Optimization to Decision Tree parameters."""
    from bayes_opt import BayesianOptimization
    def crossval_wrapper_lightgbm(num_leaves, max_depth, learning_rate, n_estimators):
        """Wrapper of Decision Tree cross validation. 
           Notice how we ensure max_depth, min_samples_leaf 
           are casted to integer before we pass them along.
        """
        return DecisionTree_CrossValidation_lightgbm(num_leaves=int(num_leaves), 
                                            max_depth=int(max_depth), 
                                            learning_rate=learning_rate, 
                                            n_estimators=int(n_estimators), 
                                            data=data, 
                                            targets=targets)

    optimizer = BayesianOptimization(f=crossval_wrapper_lightgbm, 
                                     pbounds=pars, 
                                     random_state=42, 
                                     verbose=2)
    optimizer.maximize(init_points=4, n_iter=n_iter)

    return optimizer    
def run_bayesian_opt_lightgbm(x, y, num_leaves_range, max_depth_range, learning_rate_range, n_estimators_range, iters=10):

    parameters_BayesianOptimization = {"num_leaves": num_leaves_range, 
                                    "max_depth": max_depth_range, 
                                    "learning_rate": learning_rate_range, 
                                    "n_estimators": n_estimators_range
                                    }

    BayesianOptimization = optimize_DecisionTree_lightgbm(x, 
                                                y, 
                                                parameters_BayesianOptimization, 
                                                n_iter=iters)
    print(BayesianOptimization.max)
    dict = BayesianOptimization.max['params']
    dict['max_depth'] = int(round(dict['max_depth']))
    dict['n_estimators'] = int(round(dict['n_estimators']))
    dict['num_leaves'] = int(round(dict['num_leaves']))

    return dict