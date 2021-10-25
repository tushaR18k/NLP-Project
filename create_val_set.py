import pandas as pd
import os

# DO NOT CHANGE THIS
random_seed=32
#run once to split in to training and validation set
df = pd.read_csv(os.path.join('Data/TRAINING/','training.csv'),sep="\t")

from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2, random_state=random_seed)

#save into training_split or validation_split
train.to_csv('Data/TRAINING/train_split.csv', index=False, sep='\t')
test.to_csv('Data/TRAINING/val_split.csv', index=False, sep='\t')