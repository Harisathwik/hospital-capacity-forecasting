import sys
sys.path.append('D:/Sathwik/Ayush/MLOps-Github/src')
from steps.data_pull_step import data_pull_step
import pandas as pd

print('Pulling data...')
df, manifest = data_pull_step(dataset_id='g62h-syeh', select_cols='', where_clause="date>='2020-01-01T00:00:00.000'", order_clause='date ASC', limit=5000)
print(f'Shape: {df.shape}')

# List of candidate target columns
candidates = [
    'staffed_adult_icu_bed_occupancy',
    'inpatient_beds_used',
    'adult_icu_bed_utilization',
    'total_adult_patients_hospitalized_confirmed_and_suspected_covid',
    'staffed_icu_adult_patients_confirmed_and_suspected_covid',
    'total_staffed_adult_icu_beds',
    'inpatient_beds_utilization',
    'percent_of_inpatients_with_covid'
]

print('Missingness percentage:')
for col in candidates:
    if col in df.columns:
        missing = df[col].isnull().mean() * 100
        print(f'{col}: {missing:.2f}% missing ({df[col].notnull().sum()} non-null)')
    else:
        print(f'{col}: NOT IN DATAFRAME')