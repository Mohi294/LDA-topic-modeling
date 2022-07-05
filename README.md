# LDA-topic-modeling: Using matrix and bigram-trigram
There are several ways to implement lda topic modeling, in this case the gensim package is used alongside with spacy and some data cleaning packages.

The documentations of main packages are available in following links:
- Gensim: https://radimrehurek.com/gensim/
- spaCy: https://spacy.io/api/doc
- nltk: https://www.nltk.org/

# Getting Started

gensim is a fast and reliable package for topic modeling but it may not be as accurate as we want when it is not implementing bigram-trigram so by using gensim.models.Phrases the bigram-trigram was implemented.

```python
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# !python3 -m spacy download en  # run in terminal once
def process_words(texts, stop_words=stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
    texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    texts_out = []
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    # remove stopwords once more after lemmatization
    texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]    
    return texts_out

data_ready = process_words(data_words)  # processed Text Data!
```

one of the important things in this project was getting the topic for each one of the tweets, so we started using matrix to return all topics associated with a single tweet.

one of the 
