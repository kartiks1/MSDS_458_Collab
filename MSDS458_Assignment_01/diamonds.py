import pandas as pd
import seaborn as sns
 

# load the diamonds dataset
df = sns.load_dataset('diamonds')

df.head(5)

# create a dataframe with numeric variables
df_num = df.select_dtypes(include = ['float64', 'int64'])
df_num.head(5)
