'''
PART 5: Calibration-light
- Read in data from `data/`
- Use `calibration_plot` function to create a calibration curve for the logistic regression model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
- Use `calibration_plot` function to create a calibration curve for the decision tree model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
- Which model is more calibrated? Print this question and your answer. 

Extra Credit
- Compute  PPV for the logistic regression model for arrestees ranked in the top 50 for predicted risk
- Compute  PPV for the decision tree model for arrestees ranked in the top 50 for predicted risk
- Compute AUC for the logistic regression model
- Compute AUC for the decision tree model
- Do both metrics agree that one model is more accurate than the other? Print this question and your answer. 
'''

# Import any further packages you may need for PART 5
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Calibration plot function 
def calibration_plot(y_true, y_prob, n_bins=10):
    """
    Create a calibration plot with a 45-degree dashed line.
    Parameters:
        y_true (array-like): True binary labels (0 or 1).
        y_prob (array-like): Predicted probabilities for the positive class.
        n_bins (int): Number of bins to divide the data for calibration.
    Returns:
        None
    """
    #Calculate calibration values
    bin_means, prob_true = calibration_curve(y_true, y_prob, n_bins=n_bins)
    #Create the Seaborn plot
    sns.set(style="whitegrid")
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(prob_true, bin_means, marker='o', label="Model")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Plot")
    plt.legend(loc="best")
    plt.show()

def run(df_arrests_test):
    """
    Generate calibration plots for LR and DT models and print which is more calibrated
    Parameters:
        df_arrests_test (DataFrame): Test split with pred_lr and pred_dt columns
    Returns:
        None
    """
    y_true = df_arrests_test['y']
    #calibration plots
    print("Logistic Regression Calibration Plot:")
    calibration_plot(y_true, df_arrests_test['pred_lr'], n_bins=5)
    print("Decision Tree Calibration Plot:")
    calibration_plot(y_true, df_arrests_test['pred_dt'], n_bins=5)
    #compare calibration by mean absolute error from perfect calibration
    def mean_cal_error(y_true, y_prob, n_bins=5):
        """
        Return mean absolute deviation between predicted and actual probabilities across bins
        Parameters:
            y_true (array-like): Binary labels
            y_prob (array-like): Predicted probabilities for the positive class
            n_bins (int): Number of bins for calibration curve
        Returns:
            float: Mean absolute calibration error
        """
        bin_means, prob_true = calibration_curve(y_true, y_prob, n_bins=n_bins)
        return abs(bin_means - prob_true).mean()
    lr_err = mean_cal_error(y_true, df_arrests_test['pred_lr'])
    dt_err = mean_cal_error(y_true, df_arrests_test['pred_dt'])
    more_calibrated = 'logistic regression' if lr_err < dt_err else 'decision tree'
    print(f"Which model is more calibrated? {more_calibrated} (logistic regression error: {lr_err:}, decision tree error: {dt_err:})")