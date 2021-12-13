from preprocess import preprocess
from vectorize import Glove, TFIDF

if __name__ == '__main__':
    vectorize = TFIDF(200)
    #vectorize = Glove(20,10) not working yet
    ds = preprocess("trump_train.tsv")
    X,y = vectorize.fit_transform(ds['text'],ds['device'])