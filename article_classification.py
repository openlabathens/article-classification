import pandas as pd
import numpy as np

df = pd.read_csv('data/data_set.csv')

homocides_df = df.loc[df['class'] == 'Ανθρωποκτονία']
feminicides_df = df.loc[df['class'] == 'Γυναικοκτονία']

# Get number of rows:
print(len(homocides_df.index))
print(len(feminicides_df.index))