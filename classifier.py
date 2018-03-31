
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB

df=pd.read_csv("fake_or_real_news.csv")

""" Used CountVectorizer for text classification """


# Print the head of df
print(df.head())

# Created a series to store the labels: y
y = df.label

# Created training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], y,test_size=0.33, random_state=53 )

# Initialized a CountVectorizer object: count_vectorizer
count_vectorizer = CountVectorizer(stop_words="english")

# Transformed the training data using only the 'text' column values: count_train 
count_train = count_vectorizer.fit_transform(X_train.values)

# Transformed the test data using only the 'text' column values: count_test 
count_test = count_vectorizer.transform(X_test.values)

# Print the first 10 features of the count_vectorizer
print(count_vectorizer.get_feature_names()[:10])

""" Used TfidfVectorizer for text classification (Optional) """

# Initialized a TfidfVectorizer object: tfidf_vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

tfidf_train = tfidf_vectorizer.fit_transform(X_train.values)

tfidf_test = tfidf_vectorizer.transform(X_test.values)

print(tfidf_vectorizer.get_feature_names()[:10])

# Print the first 5 vectors of the tfidf training data
print(tfidf_train.A[:5])


""" Inspected the vectors by converting them into pandas DataFrames. """

# Create the CountVectorizer DataFrame
count_df = pd.DataFrame(count_train.A, columns= count_vectorizer.get_feature_names())

# Create the TfidfVectorizer DataFrame
tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())

# Print the head of count_df
print(count_df.head())

# Print the head of tfidf_df
print(tfidf_df.head())

# Calculate the difference in columns: difference
difference = set(count_df.columns) - set(tfidf_df.columns)
print(difference)

# Check whether the DataFrames are equal
print(count_df.equals(tfidf_df))


""" Training and testing the "fake news" model with CountVectorizer """


# Instantiated a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier = MultinomialNB()

# Fit the classifier to the training data
nb_classifier.fit(count_train, y_train)

# Create the predicted tags: pred
pred = nb_classifier.predict(count_test)

# Calculate the accuracy score: score
score = metrics.accuracy_score(y_test, pred)
print(score)

# Calculate the confusion matrix: cm
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE','REAL'])
print(cm)


""" Training and testing the "fake news" model with TfidfVectorizer (Optional)"""


nb_classifier = MultinomialNB()

nb_classifier.fit(tfidf_train,y_train)

pred = nb_classifier.predict(tfidf_test)


score1 = metrics.accuracy_score(y_test, pred)
print(score1)

cm1 = metrics.confusion_matrix(y_test,pred, labels=['FAKE', 'REAL'])
print(cm1)

""" tested a few different alpha levels using the Tfidf vectors to determine if there is a better performing combination. """

#Create the list of alphas: alphas
alphas = np.arange(0,1,0.1)


def train_and_predict(alpha):
    # Instantiate the classifier: nb_classifier
    nb_classifier = MultinomialNB(alpha=alpha)
    # Fit to the training data
    nb_classifier.fit(tfidf_train, y_train)
    # Predict the labels: pred
    pred = nb_classifier.predict(tfidf_test)
    # Compute accuracy: score
    score = metrics.accuracy_score(y_test,pred)
    return score

# Iterate over the alphas and print the corresponding score
for alpha in alphas:
    print('Alpha: ', alpha)
    print('Score: ', train_and_predict(alpha))
    print()
