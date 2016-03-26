import matplotlib.pyplot as plt
import csv
from textblob import TextBlob
import pandas
import sklearn
import cPickle
import re
import string
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split 
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn import linear_model
from sklearn.learning_curve import learning_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


#splitting tweets into tokens
def split_into_tokens(message):
    message = unicode(message, 'utf8')  # convert bytes into proper unicode
    return TextBlob(message).words

#splitting tweets into lemmas
def split_into_lemmas(message):
    message = unicode(message, 'utf8').lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma
    return [word.lemma for word in words]

#counting the number of hyperlinks in tweets
def link_count(message):
    x= re.compile(r"(http://[^ ]+)").findall(message)
    return len(x)

#returns list of words other than hyperlinks in tweets
def word_without_link(message):
    x= re.compile(r"(http://[^ ]+)").findall(message)
    x1 = re.compile('\w+').findall(message)
    x2 = []
    x3 = []
    x4 = []
    for item in x:
        each_word = re.compile('\w+').findall(item)
        x2.append(each_word)
    for item2 in x2:
        for item3 in item2:
            if item3 in x1:
                x3.append(item3)
    for item4 in x1:
        if item4 not in x3:
            x4.append(item4)
    return x4

#counts the number of words without hyperlinks in tweets
def word_count_without_link(message):
    word_count = word_without_link(message)
    return len(word_count)

#counts the number of shortened URLS
def shortened_link_count(message):
    x= re.compile(r"(http://[^ ]+)").findall(message)
    count1 = 0
    x5=[]
    for item5 in x:
        each_word1 = re.compile('\w+').findall(item5)
        x5.append(each_word1)
    for item6 in x5:
        for item7 in item6:
            if (item7 == 'ly') or (item7 == 'tinyurl') or (item7 == 'it') or (item7 == 'tv') or (item7 == 'sx') or (item7 == 'gl'):
                count1 = count1 + 1
    return count1



#############################################################################################
##GETTING THE TWEET PROFILE OF EACH USER FOR TRAINING

ham_text_profile = pandas.read_csv('Training_data/legitimate_users_tweets.txt', sep='\t', quoting=csv.QUOTE_NONE,
                           names=["ID","tweetID",'Tweet','CreatedAt',"class"])
ham_text_profile['class'] = 0


spam_text_profile = pandas.read_csv('Training_data/spammers_tweets.txt', sep='\t', quoting=csv.QUOTE_NONE,
                           names=["ID","tweetID",'Tweet','CreatedAt',"class"])

spam_text_profile['class'] = 1

result1 = [ham_text_profile,spam_text_profile]
total_file_tweets = pandas.concat(result1)
#################################################################################################

################################################################################
##ADDIDITONAL COLUMNS FOR FEATURE EXTRACTION

total_file_tweets['length'] = total_file_tweets['Tweet'].map(lambda text: len(text))
total_file_tweets['http_count'] = total_file_tweets.Tweet.apply(link_count)
total_file_tweets['word_without_link'] = total_file_tweets.Tweet.apply(word_without_link)
total_file_tweets['word_count_without_link'] = total_file_tweets.Tweet.apply(word_count_without_link)
total_file_tweets['word_stripped'] = total_file_tweets['word_without_link'].map(lambda text: ' '.join(text))
total_file_tweets['shortened_link'] = total_file_tweets.Tweet.apply(shortened_link_count)

##################################################################
###Now we'll convert each message, represented as a list of tokens (lemmas) above, into a vector that machine learning models can understand.
# Doing that requires essentially three steps, in the bag-of-words model:
# counting how many times does a word occur in each message (term frequency)
# weighting the counts, so that frequent tokens get lower weight (inverse document frequency)
# normalizing the vectors to unit length, to abstract from the original text length (L2 norm)
# Here we used scikit-learn (sklearn), a powerful Python library for teaching machine learning.
# It contains a multitude of various methods and options.

bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(total_file_tweets['word_stripped'])

# The bag-of-words counts for the entire TWEETS corpus are a large, sparse matrix:

tweets_bow = bow_transformer.transform(total_file_tweets['word_stripped'])
print 'sparse matrix shape:', tweets_bow.shape
print 'number of non-zeros:', tweets_bow.nnz
print 'sparsity: %.2f%%' % (100.0 * tweets_bow.nnz / (tweets_bow.shape[0] * tweets_bow.shape[1]))


# And finally, after the counting, the term weighting and normalization can be done with TF-IDF, using scikit-learn's TfidfTransformer:

tfidf_transformer = TfidfTransformer().fit(tweets_bow)


#To transform the entire bag-of-words corpus into TF-IDF corpus at once:

messages_tfidf = tfidf_transformer.transform(tweets_bow)
print messages_tfidf.shape

###################################################################################################
###############################  TRAINING THE MODEL  #############################################
# With messages represented as vectors, we can finally train our spam/ham classifier.
# This part is pretty straightforward, and there are many libraries that realize the training algorithms.

spam_detector_bow_NB = MultinomialNB().fit(messages_tfidf, total_file_tweets['class'])

#################################### TESTING ON ITSELF  i.e on the training set############################################
# all_predictions = spam_detector.predict(messages_tfidf)
# print all_predictions
#
# print 'accuracy', accuracy_score(total_file_tweets['class'], all_predictions)
# print 'confusion matrix\n', confusion_matrix(total_file_tweets['class'], all_predictions)
# print '(row=expected, col=predicted)'

# # ################################################
# total_file_tweets.hist(column='word_without_link',by='class',bins=25)
# total_file_tweets.hist(column='http_count',by='class',bins=25)
# total_file_tweets.hist(column='length',by='class',bins=25)
# plt.show()
###############################################


####################################################################################################
######### COMBINATION OF FEATURES  ######################################################

word_c_w_l = total_file_tweets['word_count_without_link']
short_link =  total_file_tweets['shortened_link']
html_count = total_file_tweets['http_count']
S1 = pandas.Series(short_link)
S2 = pandas.Series(word_c_w_l)
S3 = pandas.Series(html_count)
df1 = pandas.DataFrame(S1)
df2 = pandas.DataFrame(S2)
df3 = pandas.DataFrame(S3)

result = pandas.concat([df1,df2,df3],axis = 1)

######################################################################
############## USING DECISION TREE TO TRAIN  ########################
classifier_tree = tree.DecisionTreeClassifier()
classifier_tree.fit(result, total_file_tweets['class'])

classifier_tree_bow = tree.DecisionTreeClassifier()
classifier_tree_bow.fit(messages_tfidf, total_file_tweets['class'])

######################################################################

######################################################################
############## USING ADA BOOST TO TRAIN  ########################
classifier_ada = AdaBoostClassifier(n_estimators=100)
classifier_ada.fit(result, total_file_tweets['class'])

classifier_ada_bow = AdaBoostClassifier(n_estimators=100)
classifier_ada_bow.fit(messages_tfidf, total_file_tweets['class'])

######################################################################

######################################################################
############## USING LOGISTIC REGRESSION TO TRAIN  ########################
classifier_logistic = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
classifier_logistic.fit(result, total_file_tweets['class'])

classifier_logistic_bow = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
classifier_logistic_bow.fit(messages_tfidf, total_file_tweets['class'])

######################################################################


######################################################################
############## USING SVM TO TRAIN  ########################
classifier_svm = SVC(cache_size=300,C=1.0,degree=2,gamma='auto',kernel='linear')
classifier_svm.fit(result, total_file_tweets['class'])

classifier_svm_bow = SVC(cache_size=300,C=1.0,degree=2,gamma='auto',kernel='linear')
classifier_svm_bow.fit(messages_tfidf, total_file_tweets['class'])

#################################################################



###########################################################################################################################
########################################### TESTING    #################################################################

#############################################################################################
##GETTING THE TWEET PROFILE OF EACH USER FOR TESTING

ham_text_profile_test = pandas.read_csv('Testing_data/legitimate_users_tweets.txt', sep='\t', quoting=csv.QUOTE_NONE,
                           names=["ID","tweetID",'Tweet','CreatedAt',"class"])
ham_text_profile_test['class'] = 0


spam_text_profile_test = pandas.read_csv('Testing_data/spammers_tweets.txt', sep='\t', quoting=csv.QUOTE_NONE,
                           names=["ID","tweetID",'Tweet','CreatedAt',"class"])

spam_text_profile_test['class'] = 1

result1_test = [ham_text_profile_test,spam_text_profile_test]
total_file_tweets_test = pandas.concat(result1_test)
###############################################################################################

################################################################################
## ADDIDITONAL COLUMNS FOR FEATURE EXTRACTION

total_file_tweets_test['length'] = total_file_tweets_test['Tweet'].map(lambda text: len(text))
total_file_tweets_test['http_count'] = total_file_tweets_test.Tweet.apply(link_count)
total_file_tweets_test['word_without_link'] = total_file_tweets_test.Tweet.apply(word_without_link)
total_file_tweets_test['word_count_without_link'] = total_file_tweets_test.Tweet.apply(word_count_without_link)
total_file_tweets_test['word_stripped'] = total_file_tweets_test['word_without_link'].map(lambda text: ' '.join(text))
total_file_tweets_test['shortened_link'] = total_file_tweets_test.Tweet.apply(shortened_link_count)

##################################################################

################################################
##### USING CLASSIFIER TO TEST ################

word_c_w_l_test = total_file_tweets_test['word_count_without_link']
short_link_test =  total_file_tweets_test['shortened_link']
html_count_test = total_file_tweets_test['http_count']
S1_test = pandas.Series(short_link_test)
S2_test = pandas.Series(word_c_w_l_test)
S3_test = pandas.Series(html_count_test)
df1_test = pandas.DataFrame(S1_test)
df2_test = pandas.DataFrame(S2_test)
df3_test = pandas.DataFrame(S3_test)

result_test = pandas.concat([df1_test,df2_test,df3_test],axis = 1)

###############################################################################################

print 'USING DECISION TREE'
#################### USING DECISION TREE TO PREDICT ########################################
predicted_labels = classifier_tree.predict(result_test)

print("Confusion Matrix")
print(confusion_matrix(total_file_tweets_test['class'], predicted_labels))
print("Precision")
print(precision_score(total_file_tweets_test['class'], predicted_labels, average=None))
print("Recall")
print(recall_score(total_file_tweets_test['class'], predicted_labels, average=None))
print("F1 score")
print(f1_score(total_file_tweets_test['class'], predicted_labels, average=None))
print 'accuracy', accuracy_score(total_file_tweets_test['class'], predicted_labels)

##############################################################################################

print 'USING ADA BOOST'
#################### USING ADA BOOST TO PREDICT ########################################
predicted_labels = classifier_ada.predict(result_test)

print("Confusion Matrix")
print(confusion_matrix(total_file_tweets_test['class'], predicted_labels))
print("Precision")
print(precision_score(total_file_tweets_test['class'], predicted_labels, average=None))
print("Recall")
print(recall_score(total_file_tweets_test['class'], predicted_labels, average=None))
print("F1 score")
print(f1_score(total_file_tweets_test['class'], predicted_labels, average=None))
print 'accuracy', accuracy_score(total_file_tweets_test['class'], predicted_labels)

##############################################################################################

print 'USING LOGISTIC REGRESSION'
#################### USING LOGISTIC REGRESSION TO PREDICT ########################################
predicted_labels = classifier_logistic.predict(result_test)

print("Confusion Matrix")
print(confusion_matrix(total_file_tweets_test['class'], predicted_labels))
print("Precision")
print(precision_score(total_file_tweets_test['class'], predicted_labels, average=None))
print("Recall")
print(recall_score(total_file_tweets_test['class'], predicted_labels, average=None))
print("F1 score")
print(f1_score(total_file_tweets_test['class'], predicted_labels, average=None))
print 'accuracy', accuracy_score(total_file_tweets_test['class'], predicted_labels)

##############################################################################################

print 'USING SUPPORT VECTOR MACHINE'
#################### USING SVM TO PREDICT ########################################
predicted_labels = classifier_svm.predict(result_test)

print("Confusion Matrix")
print(confusion_matrix(total_file_tweets_test['class'], predicted_labels))
print("Precision")
print(precision_score(total_file_tweets_test['class'], predicted_labels, average=None))
print("Recall")
print(recall_score(total_file_tweets_test['class'], predicted_labels, average=None))
print("F1 score")
print(f1_score(total_file_tweets_test['class'], predicted_labels, average=None))
print 'accuracy', accuracy_score(total_file_tweets_test['class'], predicted_labels)

##############################################################################################


print 'USing BOW predictions'
###########################################################################################
###################################  USING BOW PREDICTION ###################################

bow_transformer_test= CountVectorizer(analyzer=split_into_lemmas).fit(total_file_tweets_test['word_stripped'])
tweets_bow_test = bow_transformer_test.transform(total_file_tweets_test['word_stripped'])

tfidf_transformer_test= TfidfTransformer().fit(tweets_bow_test)
messages_tfidf_test = tfidf_transformer_test.transform(tweets_bow_test)

###############################################################################################

print 'USING DECISION TREE'
#################### USING DECISION TREE TO PREDICT ########################################
predicted_labels = classifier_tree_bow.predict(messages_tfidf_test)

print("Confusion Matrix")
print(confusion_matrix(total_file_tweets_test['class'], predicted_labels))
print("Precision")
print(precision_score(total_file_tweets_test['class'], predicted_labels, average=None))
print("Recall")
print(recall_score(total_file_tweets_test['class'], predicted_labels, average=None))
print("F1 score")
print(f1_score(total_file_tweets_test['class'], predicted_labels, average=None))
print 'accuracy', accuracy_score(total_file_tweets_test['class'], predicted_labels)

##############################################################################################

print 'USING ADA BOOST'
#################### USING ADA BOOST TO PREDICT ########################################
predicted_labels = classifier_ada_bow.predict(messages_tfidf_test)

print("Confusion Matrix")
print(confusion_matrix(total_file_tweets_test['class'], predicted_labels))
print("Precision")
print(precision_score(total_file_tweets_test['class'], predicted_labels, average=None))
print("Recall")
print(recall_score(total_file_tweets_test['class'], predicted_labels, average=None))
print("F1 score")
print(f1_score(total_file_tweets_test['class'], predicted_labels, average=None))
print 'accuracy', accuracy_score(total_file_tweets_test['class'], predicted_labels)


##############################################################################################

print 'USING LOGISTIC REGRESSION'
#################### USING LOGISTIC REGRESSION TO PREDICT ########################################
predicted_labels = classifier_logistic_bow.predict(messages_tfidf_test)

print("Confusion Matrix")
print(confusion_matrix(total_file_tweets_test['class'], predicted_labels))
print("Precision")
print(precision_score(total_file_tweets_test['class'], predicted_labels, average=None))
print("Recall")
print(recall_score(total_file_tweets_test['class'], predicted_labels, average=None))
print("F1 score")
print(f1_score(total_file_tweets_test['class'], predicted_labels, average=None))
print 'accuracy', accuracy_score(total_file_tweets_test['class'], predicted_labels)

#############################################################################################

print 'USING SUPPORT VECTOR MACHINE'
#################### USING SVM TO PREDICT ########################################
predicted_labels = classifier_svm_bow.predict(messages_tfidf_test)

print("Confusion Matrix")
print(confusion_matrix(total_file_tweets_test['class'], predicted_labels))
print("Precision")
print(precision_score(total_file_tweets_test['class'], predicted_labels, average=None))
print("Recall")
print(recall_score(total_file_tweets_test['class'], predicted_labels, average=None))
print("F1 score")
print(f1_score(total_file_tweets_test['class'], predicted_labels, average=None))
print 'accuracy', accuracy_score(total_file_tweets_test['class'], predicted_labels)
