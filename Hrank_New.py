import urllib
import nltk
import sys
import re 

import lxml
import math
import string
import textwrap
import requests

from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import defaultdict,Counter
from nltk.corpus import stopwords
from collections import defaultdict 
from bs4.element import Comment

from nltk import wordpunct_tokenize
from urllib.parse import urlparse 
import sklearn.cluster
import nltk
from nltk import word_tokenize
from nltk import word_tokenize
from nltk.corpus import wordnet as wn
import pandas as pd 
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
Common_Nouns ="january debt est dec big than who use jun jan feb mar apr may jul agust dec oct ".split(" ")
URL_CommnWords =['','https','www','com','-','php','pk','fi','http:','http']
URL_CommonQueryWords = ['','https','www','com','-','php','pk','fi','https:','http','http:']
UselessTagsText =['html','style', 'script', 'head',  '[document]','img']
def Scrapper1(element):
    if element.parent.name in [UselessTagsText]:
        return False
    if isinstance(element, Comment):
        return False
    return True

def Scrapper2(body):             
    soup = BeautifulSoup(body, 'lxml')      
    texts = soup.findAll(text=True)   
    name =soup.findAll(name=True) 
    visible_texts = filter(Scrapper1,texts)        
    return u" ".join(t.strip() for t in visible_texts)

def Scrapper3(text):                  
    lines = (line.strip() for line in text.splitlines())    
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    return u'\n'.join(chunk for chunk in chunks if chunk)


def Scrapper_title_4(URL):
  req = urllib.request.Request(URL, headers={'User-Agent' : "Magic Browser"})
  con = urllib.request.urlopen(req)
  html= con.read()
  title=[]
  
  soup = BeautifulSoup(html, 'lxml') 
  title.append(soup.title.string)
  return(title,urls)

def Web_Funtion(URL):
  req = urllib.request.Request(URL, headers={'User-Agent' : "Magic Browser"})
  con = urllib.request.urlopen(req)
  html= con.read()  
  Raw_HTML_Soup = BeautifulSoup(html, 'lxml') 
 
  raw =Scrapper2(html)
  Raw_text = Scrapper3(raw) 
  return(Raw_text,Raw_HTML_Soup) 
#----------------------------------------------------------------------------
def Preprocessing_Text(Raw_text):
    
    # 1 making text as a space seperated word list
    stopwords_nltk = set(stopwords.words("English")) 
    Words_in_text =[]
    for word in Raw_text.split():                    
        Words_in_text.append(word)

    
     #2 remove numbers and special charactes from words
        
    alphawords_only = [word for word in Words_in_text if word.isalpha()]          
    
    #3 removing length 1 words
    
    Words_afterRemoval_onelength = [word for word in alphawords_only if len(word)>1]

    #4 lower case all words
    
    lower_case_only = [word.lower() for word in Words_afterRemoval_onelength ]
    
    # Remove stopwords 
    
     
    words_withoutStopwords = [word for word in lower_case_only if word not in stopwords_nltk]
    
    #removing words from common nouns like thank, use, gift, close
    
    words_withoutCommonNouns = [word for word in words_withoutStopwords if word not in Common_Nouns ]
    
    #return list of preprocess words
    
    return (words_withoutCommonNouns)

#<hr class ="new3">
def Calc_words_frequency(Text_words):
    
    Sorted_WordCount_dict ={}  
    word_and_fr_list=[]
    Count_fr = Counter(Text_words)    
    
    for word,word_count in Count_fr.most_common():
        word_and_fr_list.append([word, word_count])
        Sorted_WordCount_dict[word]= word_count
        
    return(Sorted_WordCount_dict)
def POS_seperator(Text):
    adj=[]
    verb=[]
    nouns=[]
    for line in Text:
        tokens = nltk.word_tokenize(line)
        tagged = nltk.pos_tag(tokens)   
        
        for x,y in tagged:
            if y in ['NNP','NNPS','NNS','NN']:
                nouns.append(x)
            if y in ['JJ', 'JJR', 'JJS']:
                adj.append(x)
            if y in ['VB','VBD','VBG','VBN','VBP']:
                verb.append(x)
    return (nouns,adj,verb)


def Count_frequencies_for_POS(N,POS_text):
    Word_only=[]
    Word_frequency_only=[]
    words_and_freq = Counter(POS_text)

    for word,counts in words_and_freq.most_common(N):
        Word_only.append(word)
        Word_frequency_only.append(counts)

    return(Word_only,words_and_freq,Word_frequency_only)
#------------------------------------------------------------------------------------
#wORDnET
def Get_Synsets_Score (most_frequent_40_nouns):
    words_list_with_synsets=[]
    word_list_without_synsets =[]
    for word in most_frequent_40_nouns:
        a1 =wn.synsets(word)
        if len(a1) > 0:
            words_list_with_synsets.append(word)
        else:
            word_list_without_synsets.append(word)
    
    return (words_list_with_synsets,word_list_without_synsets)
#--------------------------------------------------------------------------------
def Get_Clusters(fr,t6,clusters_to_write):
    #f = open(clusters_to_write,'w', encoding="utf8")
 
    simstr = ""
    wordlist = []
    dm = []
    for i in t6:
        a1 =wn.synsets(i)
        a2 =(a1[0])
        dm.append([])
        wordlist.append(i)
        
        for x in t6:
            b1 =wn.synsets(x)
            b2 =([b1][0][0])                                   
            wup1 =a2.wup_similarity(b2)         
            if wup1 is None:
                simstr+="0.0 "
                dm[-1].append(1.0)
                continue        
            dm[-1].append(1.0-wup1)
            simstr += str(wup1)+" " 
        simstr += "\n"
          
    
    num_clusters=8
    agg = sklearn.cluster.AgglomerativeClustering(n_clusters=num_clusters, affinity='precomputed',linkage="complete")
    cluster_labels=agg.fit_predict(dm)
    k=[]
    m =[]
    d=[]
    for i in range(num_clusters):       
        for j in range(len(cluster_labels)):
            if cluster_labels[j] == i:
                k.append(["cluster",i])
                k.append(t6[j])
                k.append(fr[t6[j]])
                m.append(k)
            d.append(m)
    keywords = []
    clusters = {}
        
    for i in range(num_clusters):
        clusters[i] = {}
        clusters[i]['clusterSize'] = 0
        clusters[i]['items'] = []

    clusterSizes = [0] * num_clusters
    for i in range(len(cluster_labels)):
        clusters[cluster_labels[i]]['clusterSize'] += fr[t6[i]]
        clusters[cluster_labels[i]]['items'].append([t6[i], fr[t6[i]]])
        clusterSizes[cluster_labels[i]] += fr[t6[i]]

    maxClusterSize=max(clusterSizes)
    maxFrequency = fr[max(fr, key=fr.get)]

    for i in range(num_clusters):
        if clusters[i]['clusterSize'] < maxClusterSize*0.3:
            continue
        keywords.append(clusters[i]['items'][0][0])
        for word in clusters[i]['items'][1:-1]:
            if word[1] > 3 and word[1] > 0.2*maxFrequency:                    
                keywords.append(word[0])       
  
    return(keywords)


#----------------------------------------------------------------------------
def Hrank(URL):
    (Raw_text,Raw_HTML_Soup) =Web_Funtion(URL)
    preprocess_TextWords = Preprocessing_Text(Raw_text)
    text_length = len(preprocess_TextWords)  

    (nouns,adjectives,verbs) = POS_seperator(preprocess_TextWords)
    #get top frequent 40 nouns and 2 adjectives and 1 verb 
    length_nouns = len(nouns)
    preprocess_TextWords = Preprocessing_Text(Raw_text)
    text_length = len(preprocess_TextWords)     
    words_count_dic = Calc_words_frequency(preprocess_TextWords)


    (nouns,adjectives,verbs) = POS_seperator(preprocess_TextWords)
        #get top frequent 40 nouns and 2 adjectives and 1 verb 
    length_nouns = len(nouns)

    (most_frequent_40_nouns,frequencies_nouns,counts_nouns)= Count_frequencies_for_POS(40,nouns) 
    (Adjectives_two, frequencies_adjectives, count_adjective)= Count_frequencies_for_POS(2,adjectives)
    (Verb_one,frequencies_verb,count_verb) = Count_frequencies_for_POS(1,verbs)

        # two seperate list Based on the WordNet

    (words_list_with_synsets, word_list_without_synsets)= Get_Synsets_Score(most_frequent_40_nouns)
        #
    keywords = Get_Clusters(frequencies_nouns, words_list_with_synsets,"cluster.txt")

    keywords_combine =list( keywords + Adjectives_two + Verb_one)
    return keywords_combine


if __name__ == "__main__":    
    URL ="http://bbc.com"
    Keywords = Hrank(URL)
    print (Keywords)