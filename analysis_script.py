# run with: python analysis_script.py --dataset mami
import argparse
from nltk.tokenize import wordpunct_tokenize
from utils import get_part_of_speech, vectorize, plot_histogram
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import os
from pdb import set_trace as breakpoint
#from matplotlib.
class TextDataset:
    def __init__(self, path_to_data, dataset_name):
        self.path_to_data = path_to_data
        self.dataset_name = dataset_name
        self.df = self.get_dataset()
        
        self.df_with_pos= get_part_of_speech(self.df)
        print(f'Loaded the part of speech')
        self.c_vectorizer, self.df_with_pos=vectorize(self.df_with_pos)
        print(f'Finished vectorizing the part of speech')
        self.mean_pos=self.df_with_pos.mean() #per sentence
        #self.sum_pos=self.df_with_pos.sum() #per sentence
    def get_dataset(self):
        if self.dataset_name == 'mami':
            df = pd.read_csv(self.path_to_data, sep="\t")

            df['text']=df['Text Transcription'].apply(lambda x : wordpunct_tokenize(x))
            return df
        return None
    def get_ner_hist(self, args):
        doc = nlp("This is a sentence.")
        ner = nlp.add_pipe("ner")
        # This usually happens under the hood
        processed = ner(doc)
        return None
    def get_pos_hist(self, args):
        # save histogram under visualizations folder
        plt=plot_histogram(self.mean_pos.sort_values(ascending=False), bins=self.mean_pos.sort_values(ascending=False).index, x_label='Part of speech', y_label='Frequency per Sentence', bar_chart=True)
        os.makedirs(os.path.join('visualizations', self.dataset_name), exist_ok=True)
        plt.savefig(os.path.join('visualizations', self.dataset_name, 'pos.png'))
        plt.close()

    def get_average_sentence_length(self, args):
        # save histogram under visualizations folder
        self.df['sentence_counts'] = self.df['text'].apply(lambda x : len(x))
        plt=plot_histogram(self.df['sentence_counts'], bins=self.df['sentence_counts'].max())
        
        os.makedirs(os.path.join('visualizations', self.dataset_name), exist_ok=True)
        plt.savefig(os.path.join('visualizations', self.dataset_name, 'sentence_count.png'))
        plt.close()
        # return mean
        return self.df['sentence_counts'].mean()
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default='mami',
                    help='Select dataset from {mami, coco, hateful_memes, meme_gen}')
    parser.add_argument('--data_path', type=str, default='./Data/TRAINING/training.csv',
                    help='Enter path to csv file')
    args = parser.parse_args()

    # load dataset for analysis
    dataset=TextDataset(path_to_data=args.data_path, dataset_name=args.dataset)
    
    # get average sentence length
    print("Average sentence length: ", dataset.get_average_sentence_length(args))
    print("Average parts of speech per meme text: ", dataset.mean_pos.sort_values(ascending=False))
    dataset.get_pos_hist(args)