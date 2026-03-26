'''
PART 3: Logistic Regression
- Read in `df_arrests`
- Use train_test_split to create two dataframes from `df_arrests`, the first is called `df_arrests_train` and the second is called `df_arrests_test`. Set test_size to 0.3, shuffle to be True. Stratify by the outcome
- Create a list called `features` which contains our two feature names: pred_universe, num_fel_arrests_last_year
- Create a parameter grid called `param_grid` containing three values for the C hyperparameter. (Note C has to be greater than zero)
- Initialize the Logistic Regression model with a variable called `lr_model`
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv`
- Run the model
- What was the optimal value for C? Did it have the most or least regularization? Or in the middle? Print these questions and your answers.
- Now predict for the test set. Name this column `pred_lr`
- Return dataframe(s) for use in main.py for PART 4 and PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 4 and PART 5 in main.py
'''

# Import any further packages you may need for PART 3
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression as lr

def run(df_arrests):
    """
    Train a logistic regression with grid search CV and return train/test splits with pred_lr predictions
    Parameters:
        df_arrests (DataFrame): Arrests dataframe with features
    Returns:
        df_arrests_train (DataFrame): Training split
        df_arrests_test (DataFrame): Test split with pred_lr column
    """
    #train/test split stratified by outcome y
    df_arrests_train, df_arrests_test = train_test_split(df_arrests, test_size=0.3, shuffle=True, stratify=df_arrests['y'])
    #two features
    features = ['current_charge_felony', 'num_fel_arrests_last_year']
    #parameter grid for C
    param_grid = {'C': [0.01, 1.0, 100.0]}
    #initialize logistic regression and grid search
    lr_model = lr()
    gs_cv = GridSearchCV(lr_model, param_grid, cv=5)
    #fit on training set
    gs_cv.fit(df_arrests_train[features], df_arrests_train['y'])
    # Print optimal C and regularization interpretation
    best_c = gs_cv.best_params_['C']
    c_values = sorted(param_grid['C'])
    if best_c == c_values[0]:
        reg_description = 'most regularization'
    elif best_c == c_values[-1]:
        reg_description = 'least regularization'
    else:
        reg_description = 'middle regularization'
    print(f"What was the optimal value for C? {best_c}")
    print(f"Did it have the most or least regularization, or in the middle? {reg_description}")
    #predict probabilities
    df_arrests_test = df_arrests_test.copy()
    df_arrests_test['pred_lr'] = gs_cv.predict_proba(df_arrests_test[features])[:, 1]
    return df_arrests_train, df_arrests_test
