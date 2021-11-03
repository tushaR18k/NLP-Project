# For analysis
1) Move csv file to the Data folder
2) The run analysis_script using the following arguments: ```python analysis_script.py --dataset mami --data_path ./Data/TRAINING/training.csv```
3) Be sure to grab any visualizations in the visualizations folder

* if you want to do analysis on a different dataset add code under an additional if statement whereever there is 'if self.dataset_name' and make sure the text is tokenized for self.df['text']. See how its done for mami as reference.
* for the bar chart I take only the top ten {x-axis unit} but you should change this if you want more 
# For training

First run `python create_val_set.py` to get the validation and test sets. Edit the path if need be.
Change parameters in `params.py`
Then run train and validation set with `python train_script.py`
[//]: # (or `python train_script.py --model logres --max-epochs 100`


# For installation 
Installing spacy can be challenging sometimes. Follow the pip install instructions in this link: https://spacy.io/