# comp479project3
COMP 479 final project about web crawling (worth 22% for Undergrad, 20% for Grad)
## Use Scrapy for Web Crawling (5pts)
- install [Scrapy](https://doc.scrapy.org/en/latest/intro/install.html)
```
pip install Scrapy
```
- cd into project and type in cmd
```
scrapy crawl about_crawl
```
## Creating Enhanced Inverted-Index (5pts)
 - associate tf-idf and sentiment values to each term in the index

A piece of code in spimi.py starting from line 32:
```
def add_to_postings_list(postings_list, newid, term):
    """
    :param postings_list: specific postings_list from the disk
    :param newid: id of the document
    :param term: term from a token tuple
    """
    sentiment_value = 0
    term_frequency = 1
    afinn = dict(map(lambda p: (p[0], int(p[1])), [line.split('\t') for line in open("AFINN-111.txt")]))
    for key in afinn.keys():
        if key == term:
            sentiment_value = afinn[key]
            break
    for item in postings_list:
        if item[0] == newid:
            item[1] += 1
            item[3] = (1 + math.log(item[1]))*(math.log(N/len(postings_list)))
            return
    document_frequency = len(postings_list) + 1
    tf_idf_weight = (1 + math.log(term_frequency))*(math.log(N/document_frequency))
    postings_list.append([newid, term_frequency, sentiment_value, tf_idf_weight])
```
The above code provides an inverted index of the following form:

- term : [ [doc_title, tf, sentiment_value, tf_idf_weight], [posting#2], [posting#3],...]

**Enhancement Explanation**
- The initial inverted index written for the assignment 1 & 2 has the form term:postings_list
- The individual posting in the postings_list only include the docID
- Now, the individual posting include document title, the term frequency, the term sentiment value, and td-idf weight

## Develop a Sentiment Aggregation Function (3pts)
- associate sentiment to documents, queries or collections of documents

A piece of code in rank_document.py
```
def sentiment_aggr_function(some_text):
    afinn = dict(map(lambda p: (p[0], int(p[1])), [line.split('\t') for line in open("AFINN-111.txt")]))
    sentiment_aggr = sum(map(lambda word: afinn.get(word, 0), some_text.lower().split()))
    return sentiment_aggr
```
The above code takes some text (e.g. document text, query text) as argument. It then splits all the text into lowercased words,
and if some of the words appear in the AFINN list, the associated values to the words are summed up to provide the sentiment
score for the document/query/collections.

## Rank Retrieved Documents by Partial Order on (w,s) (4pts)
- w is tf-idf based cosine distance
- s is sentiment bias (e.g. A query has positive sentiment, then s1<s2 if s1 is more positive)
## Demo (1pt)
- demonstrate crawling
- demonstrate indexing
- demonstrate ranking
## Final Report (1pt for Grad, 3 pts for Undergrad)
