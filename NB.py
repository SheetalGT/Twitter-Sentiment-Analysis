import pandas as pd
import tweepy
import csv
import nltk
import re
import string
from nltk.tokenize import TweetTokenizer
from contractions_eng import contractions_dict
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk import word_tokenize, pos_tag, pos_tag_sents
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import unicodedata
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


class Preprocessor:

    def __init__(self):

        self.snowball_stemmer = SnowballStemmer("english")
        mandatory = ['no', 'not', 'against', 'never', 'nor']
        self.stopword = list(set(stopwords.words("english")) - set (mandatory))
        self.slang_dict = self.load_slang()
   
    #contractions
    def expand_contractions(self,text, contractions_dict):
        contractions_pattern = re.compile('({})'.format('|'.join(contractions_dict.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = contractions_dict.get(match) \
                                   if contractions_dict.get(match) \
                                   else contractions_dict.get(match.lower())
            expanded_contraction = expanded_contraction
            return expanded_contraction

        expanded_text = contractions_pattern.sub(expand_match, text)
        expanded_text = re.sub("'", "", expanded_text)
        return expanded_text


    #slang
    def load_slang(self):
        slangdict = dict()
        with open('slang.txt','rt') as f:
            for line in f:
                spl = line.split('\t')
                slangdict[spl[0]] = spl[1][:-1]
                
        return slangdict
    
    def remove_elongated(self,text):
        pattern = re.compile(r"(.)\1{2,}")
        return pattern.sub(r"\1\1", text)
    

    #text pre - processing
    def text_preprocessing(self,text):
        #Remove unicode characters
        # text = unicodedata.normalize('NFKD', text).encode('ascii','ignore')
        # Remove HTML special entities (e.g. &amp;)
        text = re.sub(r'\&\w*;', '', str(text))
        #Remove user handlers
        text = re.sub('@[\w]*','',text)
        # Remove tickers
        text = re.sub(r'\$\w*', '', text)
        # To lowercase
        text = text.lower()
        # Remove hyperlinks
        text = re.sub(r'https?:\/\/.*\/\w*', '', text)
        # Remove hashtags
        text = re.sub(r'#\w*', '', text)
        #Remove new lines
        text = text.replace("\n", "")
        #Remove contractions
        text= self.expand_contractions(text,contractions_dict)
        # Remove Punctuation and split 's, 't, 've with a space for filter
        text = re.sub(r'[' + string.punctuation.replace('@', '') + ']+', ' ', text)
        #Remove numbers and special characters
        #text = re.sub('[^[A-Za-z]\s]','', text)
        #text = text.replace("[^a-zA-Z]", " ")
        #Remove numbers
        text = re.sub(" \d+", " ", text)
        #Replace no with empty string
        if re.search(r'\bno\b', text):
            text = re.sub(r"\bno\b"," " ,text)
            text = re.sub(r' \b\w{1,2}\b', '', text)
            text = text + " no"
        # Remove whitespace (including new line characters)
        text = re.sub(r'\s\s+', ' ', text)
        # Remove single space remaining at the front of the tweet.
        text = text.lstrip(' ')
        # Remove characters beyond Basic Multilingual Plane (BMP) of Unicode:
        text = ''.join(c for c in text if c <= '\uFFFF')
        # Convert more than 2 letter repetitions to 2 letters
        text = text.strip('\'"?!,.():;')
        text = re.sub(r'(.)\1+', r'\1\1', text)
        # Tokenize the data
        tokens = nltk.word_tokenize(text)
        # Slang words
        tokens = [token if token not in self.slang_dict else self.slang_dict[token] for token in tokens]
        #Remove elangated words
        tokens = [self.remove_elongated(token) for token in tokens]
        #Stemmer
        stemmed_word = [self.snowball_stemmer.stem(word) for word in tokens]
        # Removing stop words
        removing_stopwords = [word for word in stemmed_word if word not in self.stopword]
        return removing_stopwords

#Main
def main():
    # Importing the dataset
    df_train = pd.read_csv("C:\python\\training.1600000.processed.noemoticon.csv ", encoding = 'latin1')
    tr = Preprocessor()
    # Text preprocessing
    df_train['TWEET_PRE'] = df_train['TWEET'].apply(tr.text_preprocessing)
    print(df_train['TWEET_PRE'].iloc[-1])
    df_train['TWEET_PRE'] = df_train['TWEET_PRE'].apply(lambda x: ' '.join(x))
    df_train['POSTags'] = pos_tag_sents(df_train['TWEET_PRE'].apply(word_tokenize).tolist())
    df_train['num_of_words'] = df_train["TWEET_PRE"].str.split().apply(len)
    print(df.head(5))
    df_train = df_train.drop(df_train[df_train.num_of_words < 1].index)
    vectorizer = CountVectorizer(ngram_range=(1,2))
    countvect_features_train= vectorizer.fit_transform(df_train.TWEET_PRE)
    #print(countvect_features)
    #tfidf values
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df= 5, norm='l2', encoding='latin-1', ngram_range=(0,1))
    features = tfidf.fit_transform(df_train.TWEET_PRE).toarray()
    labels = df_train.LABEL
    print(countvect_features.shape)
    print(labels.shape)
    clf = MultinomialNB()
    clf.fit(countvect_features,labels)
    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)

    df_test = pd.read_csv("C:\python\\testdata.manual.2009.06.14.csv ", encoding = 'latin1')
    te = Preprocessor()
    # Text preprocessing
    df_test['TWEET_PRE'] = df_test['TWEET'].apply(te.text_preprocessing)
    df_test['TWEET_PRE'] = df_test['TWEET_PRE'].apply(lambda x: ' '.join(x))
    df_test['POSTags'] = pos_tag_sents(df_test['TWEET_PRE'].apply(word_tokenize).tolist())
    df_test['num_of_words'] = df_test["TWEET_PRE"].str.split().apply(len)
    countvect_features_test= vectorizer.fit_transform(df_test.TWEET_PRE)
    #print(countvect_features)
    #tfidf values
    features = tfidf.fit_transform(df_test.TWEET_PRE).toarray()
    #print(features)
    print(countvect_features_test.shape)
    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
    emotions_predicted=clf.predict(countvect_features_test)
    print(metrics.accuracy_score(df.LABEL,emotions_predicted))
   
   
    
    

     
 
if __name__ == "__main__":
    main()
    
