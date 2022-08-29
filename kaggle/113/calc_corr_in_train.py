import pandas as pd

df = pd.read_table('train.tsv')
df = df.drop(columns=['id', 'Y'])

df_corr = df.corr()
print(df_corr[df_corr>0.5])

# Flavanoids & Total phenols                -> 0.810988
# Flavanoids & OD280/OD315 of diluted wines -> 0.758471