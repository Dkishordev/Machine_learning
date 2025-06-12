from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
import warnings

#Ignore all warnings
warnings.filterwarnings("ignore")

# Load the example dataset
dataset=[['Milk','Bread','Butter'],
        ['Bread', 'Butter'],
        ['Milk', 'Bread','Jam'],
        ['Milk','Eggs'],
        ['Bread','Eggs']]

# Convert the dataset to a pandas Dataframe
df=pd.DataFrame(dataset)

# Encode the dataset using one-hot encoding
encoded = pd.get_dummies(df.apply(pd.Series).stack()).groupby(level=0).sum()

# Generate frequent itemsets with minimum support of 0.4
frequent_itemsets = apriori(encoded, min_support=0.4, use_colnames = True)

# Generate association rules with minimum confidence of 0.4
rules = association_rules(frequent_itemsets, metric ='confidence', min_threshold =0.4)

# Print the resulting association rules
print(rules)

print(df)
print(encoded)
print(frequent_itemsets)

# Support of bread:
#4/5 = 0.8
#2/5 = 0.4