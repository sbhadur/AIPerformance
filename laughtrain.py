from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest, chi2
import data

def fit():
    df = data.clean()
    print("CLEANING SUCCESSFUL")

    X_train, X_test, y_train, y_test = train_test_split(df['clean'], df['score'], test_size = 0.2)
    pipe = Pipeline([('vect', TfidfVectorizer(ngram_range=(1,2), stop_words = "english", sublinear_tf=True)),
                     ('chi', SelectKBest(chi2, k=10000)),
                     ('clf', LinearSVC(C=1.0, penalty='l1', max_iter=3000, dual = False))])

    model = pipe.fit(X_train, y_train)

    return model
