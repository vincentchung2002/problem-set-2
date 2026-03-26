'''
PART 2: Pre-processing
- Take the time to understand the data before proceeding
- Load `pred_universe_raw.csv` into a dataframe and `arrest_events_raw.csv` into a dataframe
- Perform a full outer join/merge on 'person_id' into a new dataframe called `df_arrests`
- Create a column in `df_arrests` called `y` which equals 1 if the person was arrested for a felony crime in the 365 days after their arrest date in `df_arrests`.
- - So if a person was arrested on 2016-09-11, you would check to see if there was a felony arrest for that person between 2016-09-12 and 2017-09-11.
- - Use a print statment to print this question and its answer: What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?
- Create a predictive feature for `df_arrests` that is called `current_charge_felony` which will equal one if the current arrest was for a felony charge, and 0 otherwise.
- - Use a print statment to print this question and its answer: What share of current charges are felonies?
- Create a predictive feature for `df_arrests` that is called `num_fel_arrests_last_year` which is the total number arrests in the one year prior to the current charge.
- - So if someone was arrested on 2016-09-11, then you would check to see if there was a felony arrest for that person between 2015-09-11 and 2016-09-10.
- - Use a print statment to print this question and its answer: What is the average number of felony arrests in the last year?
- Print the mean of 'num_fel_arrests_last_year' -> pred_universe['num_fel_arrests_last_year'].mean()
- Print pred_universe.head()
- Return `df_arrests` for use in main.py for PART 3; if you can't figure this out, save as a .csv in `data/` and read into PART 3 in main.py
'''

import pandas as pd

def run():
    """
    Load raw CSVs, build features and return df_arrests
    Parameters:
        None
    Returns:
        df_arrests (DataFrame): pred_universe with target and predictive features
    """
    pred_universe = pd.read_csv('data/pred_universe_raw.csv', parse_dates=['arrest_date_univ'])
    arrest_events = pd.read_csv('data/arrest_events_raw.csv', parse_dates=['arrest_date_event'])
    #Full outer join on person_id (arrest_id conflict: pred_universe keeps '', arrest_events gets '_event')
    df_arrests = pred_universe.merge(arrest_events, on='person_id', how='outer', suffixes=('', '_event'))
    #days between each arrest_event and the index arrest date
    delta = (df_arrests['arrest_date_event'] - df_arrests['arrest_date_univ']).dt.days
    felony = df_arrests['charge_degree'] == 'felony'
    #1 if any felony rearrest within 1-365 days after index arrest
    future_felony = df_arrests[(delta >= 1) & (delta <= 365) & felony].groupby('arrest_id').size().gt(0).astype(int)
    pred_universe['y'] = pred_universe['arrest_id'].map(future_felony).fillna(0).astype(int)
    print(f"What share of arrestees were rearrested for a felony in the next year? {pred_universe['y'].mean():}")
    #1 if the current arrest is a felony
    current_charge = arrest_events.set_index('arrest_id')['charge_degree'].eq('felony').astype(int)
    pred_universe['current_charge_felony'] = pred_universe['arrest_id'].map(current_charge).fillna(0).astype(int)
    print(f"What share of current charges are felonies? {pred_universe['current_charge_felony'].mean():}")
    #count of felony arrests in the 365 days prior to index arrest
    prior_felonies = df_arrests[(delta >= -365) & (delta <= -1) & felony].groupby('arrest_id').size()
    pred_universe['num_fel_arrests_last_year'] = pred_universe['arrest_id'].map(prior_felonies).fillna(0).astype(int)
    print(f"What is the average number of felony arrests in the last year? {pred_universe['num_fel_arrests_last_year'].mean():}")
    print(pred_universe.head())
    df_arrests = pred_universe
    return df_arrests
 