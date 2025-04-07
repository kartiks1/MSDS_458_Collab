import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
 

# load the diamonds dataset
df = sns.load_dataset('diamonds')

df.head(5)

# create a dataframe with numeric variables
df_num = df.select_dtypes(include = ['float64', 'int64'])
df_num.head(5)
print(df_num.head(5))