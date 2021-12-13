# run with: python analysis_script.py --dataset mami
import argparse
from nltk.tokenize import wordpunct_tokenize
from utils import get_part_of_speech, vectorize, plot_histogram
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, namedtuple
import os
from pdb import set_trace as breakpoint
import spacy
import json
import ast
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
        elif self.dataset_name == 'coco':
            with open(self.path_to_data) as f:
                json_data = json.load(f)
                text=[datapoint['caption'] for datapoint in json_data['annotations']]
                df = pd.DataFrame()
                df['text']=text
                df['text']=df['text'].apply(lambda x : wordpunct_tokenize(x))
                return df.sample(10000)
        elif self.dataset_name == 'fb':
            with open(self.path_to_data) as f:
                texts=[]
                for line in f:
                    texts.append(ast.literal_eval(line)['text'])
                df = pd.DataFrame()
                df['text']=texts
                df['text']=df['text'].apply(lambda x : wordpunct_tokenize(x))           
                return df
            
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
<<<<<<< HEAD
        self.df_with_pos['sentence_counts']=self.df['sentence_counts']
=======
>>>>>>> da5de8aaeb66abdbb2af19c3efed604ac65990ea
        plt=plot_histogram(self.df['sentence_counts'], bins=list(range(1,300, 10)),x_label='Text Word Length', y_label='Density')
        
        os.makedirs(os.path.join('visualizations', self.dataset_name), exist_ok=True)
        plt.savefig(os.path.join('visualizations', self.dataset_name, 'sentence_count.png'))
        plt.close()
        # return mean
        return self.df['sentence_counts'].mean()
    
    def get_ner_hist(self, args):
        nlp = spacy.load("en_core_web_sm")
        def get_ner(x):
            doc = nlp(' '.join(x))
            nouns = [chunk.text for chunk in doc.noun_chunks]
            verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
            all_entities=[]
            for entity in doc.ents:
                all_entities.append(entity.label_)
            return nouns, verbs, all_entities
        self.df['noun_phrases'], self.df['verb_phrases'], self.df['ner']=zip(*self.df['text'].apply(lambda x: get_ner(x)))

        # get counts
        all_entities=[]
        all_np = []
        all_vp = []
        for np, vp, ner in zip(self.df['noun_phrases'].values, self.df['verb_phrases'].values, self.df['ner'].values):
            all_np.extend(np)
            all_vp.extend(vp)
            all_entities.extend(ner)
        obj=namedtuple(typename='test', field_names=['index', 'values'])
        counter_all_entities=zip(*Counter(all_entities).most_common())
        #breakpoint()
        obj.index, obj.values = (next(counter_all_entities), next(counter_all_entities))
        # normalize counts
        total = sum(Counter(all_entities).values(), 0.0)
        #breakpoint()
        obj.values=[val/total for val in obj.values]
        plt=plot_histogram(obj, x_label='Entities', y_label='Percent Over all Mentions', bar_chart=True)
        plt.tight_layout()
#plt.gcf().subplots_adjust(bottom=0.80)
        plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees

        os.makedirs(os.path.join('visualizations', self.dataset_name), exist_ok=True)
        plt.savefig(os.path.join('visualizations', self.dataset_name, 'ner_count.png'))
        plt.close()
        
        return Counter(all_np).most_common(20), Counter(all_vp).most_common(20), Counter(all_entities).most_common(20)
    
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
<<<<<<< HEAD
    dataset.df_with_pos.to_csv(f'results/{args.dataset}.csv')
    exit(1)

    print("Average parts of speech per meme text: ", dataset.mean_pos.sort_values(ascending=False))
    dataset.get_pos_hist(args)

=======
    exit(1)
    print("Average parts of speech per meme text: ", dataset.mean_pos.sort_values(ascending=False))
    dataset.get_pos_hist(args)
    
>>>>>>> da5de8aaeb66abdbb2af19c3efed604ac65990ea
        #todo add normalization
    np, vp, ent_a = dataset.get_ner_hist(args)
    print("Nouns, Verbs, Entities Counts: \n" )
    print("NP\n", np)
    print("VP\n", vp)
    print("Entities\n", ent_a)