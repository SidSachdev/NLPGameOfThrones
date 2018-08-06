import nltk
from nltk import FreqDist
from nltk.collocations import *
import re
#from nltk.corpus import stopwords

def alpha_filter(w):
  # pattern to match word of non-alphabetical characters
  pattern = re.compile('^[^a-z]+$')
  if (pattern.match(w)):
    return True
  else:
    return False


#Harry Potter Text initialization

Potter = open('/Users/Sidd/PycharmProjects/nltkclass/harrypotter.txt')
Pottertext = Potter.read()

#Harry Potter Text Tokenization

Pottertokens = nltk.word_tokenize(Pottertext)
Ptext = nltk.Text(Pottertokens)

Potter.close()

#Game of Thrones Text initialization

GameOT = open('/Users/Sidd/PycharmProjects/nltkclass/gameofthrones.txt')
GameOTtext = GameOT.read()

#Game Of Thrones Text Tokenization

GameOTtokens = nltk.word_tokenize(GameOTtext)
GOTtext = nltk.Text(GameOTtokens)

GameOT.close()
print('----------------------------------------------------------------------------------------------------------------')
print('----------------------------------------------------------------------------------------------------------------')
print('Number of Tokens')
print(len(Pottertokens))
print(len(GameOTtokens))
print('----------------------------------------------------------------------------------------------------------------')
print('----------------------------------------------------------------------------------------------------------------')

#Talk about size of each novel

#HP - choose to treat upper and lower case the same
#    by putting all tokens in lower case
Potterwords = [w.lower() for w in Pottertokens]

#GameOT - choose to treat upper and lower case the same
#    by putting all tokens in lower case
GameOTwords = [z.lower() for z in GameOTtokens]



print('Sorted Set')
#HP - Sort the sets
Pottervocab = sorted(set(Potterwords))
print(Pottervocab[:20])

print('----------------------------------------------------------------------------------------------------------------')

#GameOT - Sort the sets
GameOTvocab = sorted(set(GameOTwords))
print(GameOTvocab[:20])

print('----------------------------------------------------------------------------------------------------------------')
print('----------------------------------------------------------------------------------------------------------------')


#stopwords = nltk.corpus.stopwords.words('english')
#print(stopwords)

print('Without Stop Words top 20')

fstop = open('Smart.English.stop', 'r')
stoptext = fstop.read()
fstop.close()

stopwords = nltk.word_tokenize(stoptext)
#print ("Display first 50 Stopwords:")
#print (stopwords[:20])

#Potter stop words removed
Potterfiltered_words = [w for w in Potterwords if w not in stoptext]
print(Potterfiltered_words[:20])

#GameOT stop words removed
GameOTfiltered_words = [w for w in GameOTwords if w not in stoptext]
print(GameOTfiltered_words[:20])


print('----------------------------------------------------------------------------------------------------------------')
print('----------------------------------------------------------------------------------------------------------------')



#Frequency Distribution
#Harry Potter

Potterfdist = FreqDist(Potterfiltered_words)
Potterfdistkeys = list(Potterfdist.keys())
#print(Potterfdistkeys[:50])

#print(Potterfdist['Harry'])

#Game of Thrones
GameOTfdist = FreqDist(GameOTfiltered_words)
GameOTfdistkeys = list(GameOTfdist.keys())
#print(GameOTfdistkeys[:50])

#print(GameOTfdist['Tyrion'])


#Top 50 in HP
#print('Top 50 Unfiltered')
Pottertopkeys = Potterfdist.most_common(50)
#print(Pottertopkeys)

#Top 50 in GameOT

GameOTtopkeys = GameOTfdist.most_common(50)
#print(GameOTtopkeys)


print('Top 50 Filtered in Order')
for pair1 in Pottertopkeys:
    print(pair1)

print('----------------------------------------------------------------------------------------------------------------')

for pair2 in GameOTtopkeys:
    print(pair2)

print('----------------------------------------------------------------------------------------------------------------')
print('----------------------------------------------------------------------------------------------------------------')

print('Bigrams First 50')

#Potter Bigrams
Potterbigrams = list(nltk.bigrams(Potterfiltered_words))
#print(Potterbigrams[:50])

#GameOT Bigrams

GameOTbigrams = list(nltk.bigrams(GameOTfiltered_words))
#print(GameOTbigrams[:50])

print('--------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------')
bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(GameOTfiltered_words)
scored = finder.score_ngrams(bigram_measures.raw_freq)
finder.apply_word_filter(alpha_filter)
scored = finder.score_ngrams(bigram_measures.raw_freq)
for bscore in scored[:30]:
    print (bscore)
print('--------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------')
#BEFORE STOP WORDS
#for bscore in scored[:30]:
#    print (bscore)

#Bigram Freq
#finder.apply_word_filter(lambda w: w in stoptext)
#scored = finder.score_ngrams(bigram_measures.raw_freq)
#for bscore in scored[:50]:
#    print (bscore)
#print('--------------------------------------------------------------------------------')

#Remove low frequency words - Good
finder2 = BigramCollocationFinder.from_words(GameOTfiltered_words)
finder2.apply_word_filter(lambda w: w in stoptext)
finder2.apply_freq_filter(2)
finder2.apply_word_filter(alpha_filter)
scored2 = finder2.score_ngrams(bigram_measures.raw_freq)
for bscore in scored2[:50]:
    print (bscore)


print('--------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------')

# apply a filter on both words of the ngram - similar

#finder3 = BigramCollocationFinder.from_words(Potterfiltered_words)
#finder3.apply_ngram_filter(lambda w1, w2: len(w1) < 2)
#scored = finder3.score_ngrams(bigram_measures.raw_freq)
#for bscore in scored[:50]:
#    print (bscore)
print('--------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------')

# pointwise mutual information
finder4 = BigramCollocationFinder.from_words(GameOTfiltered_words)
finder4.apply_word_filter(alpha_filter)
scored = finder4.score_ngrams(bigram_measures.pmi)
for bscore in scored[:50]:
    print (bscore)
print('--------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------')
print('now in freq')
print('--------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------')



# bigrams in order by Pointwise Mutual Information.
scored = finder4.score_ngrams(bigram_measures.pmi)
finder4.apply_word_filter(alpha_filter)

for bscore in scored[:50]:
    print (bscore)

print('--------------------------------------------v-------------------------------------------------------')

finder4.apply_freq_filter(5)
finder4.apply_word_filter(alpha_filter)

scored = finder4.score_ngrams(bigram_measures.pmi)
for bscore in scored[:50]:
    print (bscore)

print('--------------------------------------------v-------------------------------------------------------')
print('--------------------------------------------v-------------------------------------------------------')




#Potter Trigrams
print('Potter Trigrams Top 50')

Pottertrigrams = list(nltk.trigrams(Potterfiltered_words))
print(Pottertrigrams[:50])

#GameOT Trigrams

print('GameOT Trigrams Top 50')

GameOTtrigrams = list(nltk.trigrams(GameOTfiltered_words))
print(GameOTtrigrams[:50])


trigram_measures = nltk.collocations.TrigramAssocMeasures()
trifinder = TrigramCollocationFinder.from_words(GameOTfiltered_words)
trifinder.apply_word_filter(alpha_filter)
triscored = trifinder.score_ngrams(trigram_measures.raw_freq)

trifinder = TrigramCollocationFinder.from_words(GameOTfiltered_words)
trifinder.apply_word_filter(lambda w: w in stopwords)
trifinder.apply_freq_filter(2)
trifinder.apply_word_filter(alpha_filter)

triscored = trifinder.score_ngrams(trigram_measures.raw_freq)
for triscore in triscored[:50]:
    print (triscore)
