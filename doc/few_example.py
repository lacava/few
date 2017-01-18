from few import FEW
import pandas as pd
from sklearn.model_selection import train_test_split

dataset = 'd_enc.txt'

input_data = pd.read_csv(dataset,sep=None,engine='python')

#generate train/test split
train_i, test_i = train_test_split(input_data.index, train_size=0.75, test_size=0.25)

# training data
X_train = input_data.loc[train_i].drop('label', axis=1).values
Y_train = input_data.loc[train_i, 'label'].values

#testing data
X_test = input_data.loc[test_i].drop('label', axis=1).values
Y_test = input_data.loc[test_i, 'label'].values

few = FEW(verbosity=1)
few.fit(X_train,Y_train)

print('\nTraining accuracy: {}'.format(few.score(X_train, Y_train)))
print('Holdout accuracy: {}'.format(few.score(X_test, Y_test)))
print('\Model: {}'.format(few.print_model()))
