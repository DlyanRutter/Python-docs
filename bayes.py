# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 18:00:56 2016

@author: dylanrutter
"""
import numpy as np

sample_memo = '''
Milt, we're gonna need to go ahead and move you downstairs into storage B. We \
have some new people coming in, and we need all the space we can get. So if you \
could just go ahead and pack up your stuff and move it down there, that would \
be terrific, OK? Oh, and remember: next Friday... is Hawaiian shirt day. So, you\
 know, if you want to, go ahead and wear a Hawaiian shirt and jeans. Oh, oh, and\
 I almost forgot. Ahh, I'm also gonna need you to go ahead and come in on Sunday,\
 too... Hello Peter, whats happening? Ummm, I'm gonna need you to go ahead\
 and come in tomorrow. So if you could be here around 9 that would be great,\
 mmmk... oh oh! and I almost forgot ahh, I'm also gonna need you to go ahead \
 and come in on Sunday too, kay. We ahh lost some people this week and ah, \
 we sorta need to play catch up.
'''

corrupted_memo = '''
Yeah, I'm gonna --- you to go ahead --- --- complain about this. Oh, and if you\
 could --- --- and sit at the kids' table, that'd be --- 
'''

def NextWordProbability(sampletext,word):
    """Takes in a sample text and a word and returns a dictionary with keys that
    are the set of words that come after the input word, whose values are the 
    number of times the key follows the input word."""
    index = 0
    words = sampletext.split()
    next_words = []
    word_counts = []
    while index <= len(words)-1 and max(words) != word:
        if words[index] == word:
            next_words.append(words[index+1])
        index+=1
    for e in next_words:
        word_counts.append(next_words.count(e))                  
    return dict(zip(next_words,word_counts))

words_to_guess = ["ahead","could"]

def LaterWords(sample,word,distance):
    '''sample: a sample of text to draw from
    word: a word occuring before a corrupted sequence
    distance: how many words later to estimate 
        (i.e. 1 for the next word, 2 for the word after that)
    returns: a single word which is the most likely possibility
    '''
    def max_probability_tuple(sample_text, chosen_word):
        dictionary = NextWordProbability(sample, chosen_word)
        overall_sum = sum(dictionary.values())
        for key in dictionary.keys():
            dictionary[key] = np.round(float(dictionary[key])/float(overall_sum), decimals=3)
        for key, value in dictionary.items():
            if value == max(dictionary.values()):
                return (key,value)    
    best_list = []
    index = 1
    while index <= distance:
        best_list.append(max_probability_tuple(sample, word))
        word = max_probability_tuple(sample, word)[0]
        index+=1    
    return best_list[len(best_list)-1][0]

#overall_sum = sum( NextWordProbability(sample_memo, 'you').values())
  
#print overall_sum  
print LaterWords(sample_memo,"and",2)            
            
    
print NextWordProbability(sample_memo, 'you')
            
print list()




