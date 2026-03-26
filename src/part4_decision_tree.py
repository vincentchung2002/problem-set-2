'''
PART 4: Decision Trees
- Read in the dataframe(s) from PART 3
- Create a parameter grid called `param_grid_dt` containing three values for tree depth. (Note C has to be greater than zero)
- Initialize the Decision Tree model. Assign this to a variable called `dt_model`.
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv_dt`.
- Run the model
- What was the optimal value for max_depth?  Did it have the most or least regularization? Or in the middle?
- Now predict for the test set. Name this column `pred_dt`
- Save dataframe(s) save as .csv('s) in `data/`
'''

# Import any further packages you may need for PART 4
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier as DTC

def run(df_arrests_train, df_arrests_test):
    """
    Train a decision tree with grid search CV, add pred_dt to test set, save CSVs, and return splits
    Parameters:
        df_arrests_train (DataFrame): Training split from Part 3
        df_arrests_test (DataFrame): Test split from Part 3
    Returns:
        df_arrests_train (DataFrame): Training split saved to data/
        df_arrests_test (DataFrame): Test split with pred_dt column added, saved to data/
    """
    features = ['current_charge_felony', 'num_fel_arrests_last_year']
    #parameter grid for max_depth
    param_grid_dt = {'max_depth': [3, 5, 10]}
    #initialize decision tree and grid search
    dt_model = DTC()
    gs_cv_dt = GridSearchCV(dt_model, param_grid_dt, cv=5)
    #fit on training set
    gs_cv_dt.fit(df_arrests_train[features], df_arrests_train['y'])
    #print optimal max_depth and regularization interpretation
    best_depth = gs_cv_dt.best_params_['max_depth']
    depth_values = sorted(param_grid_dt['max_depth'])
    if best_depth == depth_values[0]:
        reg_description = 'most regularization'
    elif best_depth == depth_values[-1]:
        reg_description = 'least regularization'
    else:
        reg_description = 'middle regularization'
    print(f"What was the optimal value for max_depth? {best_depth}")
    print(f"Did it have the most or least regularization, or in the middle? {reg_description}")
    #predict probabilities 
    df_arrests_test = df_arrests_test.copy()
    df_arrests_test['pred_dt'] = gs_cv_dt.predict_proba(df_arrests_test[features])[:, 1]
    #save to data/
    df_arrests_train.to_csv('data/df_arrests_train.csv', index=False)
    df_arrests_test.to_csv('data/df_arrests_test.csv', index=False)
    return df_arrests_train, df_arrests_test
