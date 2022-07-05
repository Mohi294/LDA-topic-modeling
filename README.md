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

One of the important things in this project was getting the topic for each one of the tweets, so we started using matrix to return all topics associated with a single tweet; also choosing an optimal number of topics is an important concern, in this case we can use ```gensim.models.coherencemodel.CoherenceModel```  inside a loop to achieve the best number of topics.

```python
LDA = gensim.models.ldamodel.LdaModel
dict_ = corpora.Dictionary(data_words)
doc_term_matrix=[dict_.doc2bow(i) for i in data_ready]
coherence = []
for k in range(2,6):
    ldaModel = LDA(doc_term_matrix, num_topics=k, id2word=dict_,chunksize=1000, passes=1, random_state=0, update_every=1, eval_every=None)
        
    cm = gensim.models.coherencemodel.CoherenceModel(model=ldaModel, texts=data_ready, dictionary=dict_,
     coherence='c_v')
    coherence.append((k,cm.get_coherence()))
x_val = [x[0] for x in coherence]
y_val = [x[1] for x in coherence]
maxCoh = max(y_val)

NumOfTopics = 0
for i in range(4):
    if y_val[i] == maxCoh:
        NumOfTopics = i+1

print(NumOfTopics)

```
# Gensim lda model parameters
for more accurate explanations visit this [page](https://radimrehurek.com/gensim/models/ldamodel.html).

alpha and beta are two parameters that were not well discussed, as it is said in this [topic](https://datascience.stackexchange.com/questions/199/what-does-the-alpha-and-beta-hyperparameters-contribute-to-in-latent-dirichlet-a) a low alpha value places more weight on having each document composed of only a few dominant topics (whereas a high value will return many more relatively dominant topics). Similarly, a low beta value places more weight on having each topic composed of only a few dominant words.

for more information on alpha and beta visit this [page](https://www.thoughtvector.io/blog/lda-alpha-and-beta-parameters-the-intuition/).

# Requirements
Python 3.6+ is required. The following packages are required:
- [Numpy](https://numpy.org/doc/)
- [Pandas](https://pandas.pydata.org/docs/)
- [logging](https://docs.python.org/3/library/logging.html)

# Notes
Latent Dirichlet allocation is described in [Blei et al. (2003)](https://jmlr.org/papers/v3/blei03a.html) and [Pritchard et al. (2000)](https://academic.oup.com/genetics/article/155/2/945/6048111). 











