import pandas as pd
import numpy as np
import seaborn as sns  
import matplotlib.pyplot as plt

#for text pre-processing
import re, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('stopwords')

#for model-building
from sklearn.model_selection import train_test_split



#LEMMATIZATION
# Initialize the lemmatizer
wl = WordNetLemmatizer()

dataframe = pd.read_csv('bbc-news-data.csv', sep='\t')
train_dataframe, test_dataframe = train_test_split(dataframe, test_size=0.2, random_state=42)


def EDA(dataframe,train_dataframe,test_dataframe):
    print(dataframe.info(),"\n\n")
    df_numeric=dataframe.describe(include=np.number)
    print(df_numeric,"\n")
    
    
    x=dataframe["Category"].value_counts()
    #print(x)
    sns.barplot(x=x.index, y=x.values)
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.title("Value Counts of Categories total data")
    plt.show()
    #test data
    x=train_dataframe["Category"].value_counts()
    sns.barplot(x=x.index, y=x.values)
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.title("Value Counts of Categories train")
    plt.show()

    #train data
    x=test_dataframe["Category"].value_counts()
    sns.barplot(x=x.index, y=x.values)
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.title("Value Counts of Categories test")
    plt.show()
    #missing values 
    print("Missing Value Count :")
    print(dataframe.isna().sum())
    #mean word count
    # WORD-COUNT
    dataframe['word_count'] = dataframe['Content'].apply(lambda x: len(str(x).split()))
    print(dataframe.head)
    print("\nMean word count of Football Article: ",dataframe[dataframe['Category']=='business']['word_count'].mean()) #Disaster tweets
    print("Mean word count of Business Article: ",dataframe[dataframe['Category']=='entertainment']['word_count'].mean()) #Non-Disaster tweets
    print("Mean word count of Politics Article: ",dataframe[dataframe['Category']=='politics']['word_count'].mean())
    print("Mean word count of Film Article: ",dataframe[dataframe['Category']=='sport']['word_count'].mean())
    print("Mean word count of Technology Article: ",dataframe[dataframe['Category']=='tech']['word_count'].mean())


#convert to lowercase, strip and remove punctuations
def preprocess(Content):
    # lowercase of letters
    Content = Content.lower() 
    #strip spaces
    Content=Content.strip()  
    #get all the special symbols
    Content=re.compile('<.*?>').sub('', Content) 
    #get format characters
    Content = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', Content)  
    #substitutions
    Content = re.sub('\s+', ' ', Content)  
    Content = re.sub(r'\[[0-9]*\]',' ',Content) 
    Content=re.sub(r'[^\w\s]', '', str(Content).lower().strip())
    Content = re.sub(r'\d',' ',Content) 
    Content = re.sub(r'\s+',' ',Content) 
    return Content

 
# STOPWORD REMOVAL
def stopword(string):
    a= [i for i in string.split() if i not in stopwords.words('english')]
    return ' '.join(a)

 
# This is a helper function to map NTLK position tags
"""WordNet is a large lexical database of English words that can be used to determine the lemma of a word. NLTK provides a WordNet lemmatizer that can be used to lemmatize words based on their part of speech (POS)."""
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
# Tokenize the sentence
"""
A lemmatizer is a natural language processing (NLP) tool that reduces words to their base forms, also known as lemmas. This process is called lemmatization. Lemmatization is similar to stemming, but it is more sophisticated and takes into account the context of the word to determine the correct lemma.

For example, the lemma of the word "running" is "run". This is because "running" is a verb form of the word "run". Similarly, the lemma of the word "saw" is "see". This is because "saw" is a past tense form of the word "see"."""
def lemmatizer(string):
    word_pos_tags = nltk.pos_tag(word_tokenize(string)) # Get position tags
    a=[wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] # Map the position tag and lemmatize the word/token
    #print ("\n\n\n\n" ,a, "\n\n\n\n")
    return " ".join(a)
def finalpreprocess(string):
    return lemmatizer(stopword(preprocess(string)))
if __name__ == "__main__":
    #print(train_dataframe.head)
    EDA(dataframe,train_dataframe,test_dataframe)
    train_dataframe['clean_text'] = train_dataframe['Content'].apply(lambda x: finalpreprocess(x))
    #print(train_dataframe.head)
    train_dataframe.to_csv('testclean.csv',sep='\t',index=False)
    train_dataframe['word_count'] = train_dataframe['clean_text'].apply(lambda x: len(str(x).split()))
    #print(dataframe.head)
    print("\nMean word count of cleaned Football Article: ",train_dataframe[train_dataframe['Category']=='business']['word_count'].mean()) #Disaster tweets
    print("Mean word count of cleaned Business Article: ",train_dataframe[train_dataframe['Category']=='entertainment']['word_count'].mean()) #Non-Disaster tweets
    print("Mean word count of cleaned Politics Article: ",train_dataframe[train_dataframe['Category']=='politics']['word_count'].mean())
    print("Mean word count of cleaned Film Article: ",train_dataframe[train_dataframe['Category']=='sport']['word_count'].mean())
    print("Mean word count of cleaned Technology Article: ",train_dataframe[train_dataframe['Category']=='tech']['word_count'].mean())
    
    
    

