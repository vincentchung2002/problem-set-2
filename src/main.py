'''
You will run this problem set from main.py, so set things up accordingly
'''

import pandas as pd
import part1_etl
import part2_preprocessing
import part3_logistic_regression
import part4_decision_tree
import part5_calibration_plot

# Call functions / instanciate objects from the .py files
def main():
    # PART 1: Instanciate etl, saving the two datasets in `./data/`
    part1_etl.run()
    # PART 2: Call functions/instanciate objects from preprocessing
    df_arrests = part2_preprocessing.run()
    # PART 3: Call functions/instanciate objects from logistic_regression
    df_arrests_train, df_arrests_test = part3_logistic_regression.run(df_arrests)
    # PART 4: Call functions/instanciate objects from decision_tree
    df_arrests_train, df_arrests_test = part4_decision_tree.run(df_arrests_train, df_arrests_test)
    # PART 5: Call functions/instanciate objects from calibration_plot
    part5_calibration_plot.run(df_arrests_test)

if __name__ == "__main__":
    main()