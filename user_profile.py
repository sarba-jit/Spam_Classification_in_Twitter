import matplotlib.pyplot as plt
import csv
from textblob import TextBlob
import pandas
import statistics as stat
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


def avg_fol_count(series):
    x= re.compile(r'\d+').findall(series)
    len_x = float(len(x))
    min_x = float(min(x))
    max_x = float(max(x))
    avg_count = (max_x+min_x)/len_x
    return avg_count

def avg_fol_count_test(series):
    x= re.compile(r'\d+').findall(series)
    len_x = float(len(x))
    min_x = float(min(x))
    max_x = float(max(x))
    avg_count = (max_x+min_x)/len_x
    return avg_count

def var_fol_count(series):
    x1 = re.compile(r'\d+').findall(series)
    x2= []
    for item in x1:
        x2.append(int(item))
    return stat.variance(x2)

def var_fol_count_test(series):
    x1 = re.compile(r'\d+').findall(series)
    x2= []
    for item in x1:
        x2.append(int(item))
    return stat.variance(x2)

#########################################################################################
##GETTING THE PROFILE OF EACH USERS FOR TRAINING

ham_profile = pandas.read_csv('Training_data/legitimate_users.txt', sep='\t', quoting=csv.QUOTE_NONE,
                           names=["ID","Created_At","Collected_At","NumberOfFollowings", "NumberOfFollowers", "NumberOfTweets", "LengthOfScreenName", "LengthOfDescriptionInUserProfile","class"])
ham_profile['class'] = 0

spam_profile = pandas.read_csv('Training_data/spammers.txt', sep='\t', quoting=csv.QUOTE_NONE,
                           names=["ID","Created_At","Collected_At","NumberOfFollowings", "NumberOfFollowers", "NumberOfTweets", "LengthOfScreenName", "LengthOfDescriptionInUserProfile","class"])

spam_profile['class'] = 1

result_profile = [ham_profile,spam_profile]
total_file_profile = pandas.concat(result_profile)
# print total_file.describe()
############################################################################################


#########################################################################################
##GETTING THE NUMBER OF FOLLOWERS FOR TRAINING

ham_following = pandas.read_csv('Training_data/legitimate_users_followings.txt', sep='\t', quoting=csv.QUOTE_NONE,
                           names=["ID",'SeriesOfNumberOfFollowings',"class"])
ham_following['class'] = 0

spam_following = pandas.read_csv('Training_data/spammers_followings.txt', sep='\t', quoting=csv.QUOTE_NONE,
                           names=["ID",'SeriesOfNumberOfFollowings',"class"])

spam_following['class'] = 1

result_following = [ham_following,spam_following]
total_file_following = pandas.concat(result_following)
# print total_file_following.describe()
############################################################################################

################################################################################
## ADDIDITONAL COLUMNS FOR FEATURE EXTRACTION

total_file_following['Avg_follower_count'] = total_file_following.SeriesOfNumberOfFollowings.apply(avg_fol_count)
total_file_following['Var_follower_count'] = total_file_following.SeriesOfNumberOfFollowings.apply(var_fol_count)


# print total_file_following.describe()
# print total_file_following['Var_follower_count'].tail()

# ##################################################################


####################################################################################################
######### COMBINATION OF FEATURES  ######################################################

avg_fol_count = total_file_following['Avg_follower_count']
var_fol_count = total_file_following['Var_follower_count']
length_profile_desc = total_file_profile['LengthOfDescriptionInUserProfile']

S1 = pandas.Series(avg_fol_count)
S2 = pandas.Series(var_fol_count)
S3 = pandas.Series(length_profile_desc)
df1 = pandas.DataFrame(S1)
df2 = pandas.DataFrame(S2)
df3 = pandas.DataFrame(S3)

result = pandas.concat([df1,df2,df3],axis = 1)

######################################################################
############## USING DECISION TREE TO TRAIN  ########################
classifier_tree = tree.DecisionTreeClassifier()
classifier_tree.fit(result, total_file_profile['class'])

######################################################################

######################################################################
############## USING ADA BOOST TO TRAIN  ########################
classifier_ada = AdaBoostClassifier(n_estimators=100)
classifier_ada.fit(result, total_file_profile['class'])

######################################################################

######################################################################
############## USING LOGISTIC REGRESSION TO TRAIN  ########################
classifier_logistic = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
classifier_logistic.fit(result, total_file_profile['class'])

######################################################################

######################################################################
############## USING SVM TO TRAIN  ########################
classifier_svm = SVC(cache_size=300,C=1.0,degree=2,gamma='auto',kernel='linear')
classifier_svm.fit(result, total_file_profile['class'])

########################################################################################################################
#########################################  TESTING  ####################################################################

#########################################################################################
##GETTING THE PROFILE OF EACH USERS FOR TESTING

ham_profile_test = pandas.read_csv('Testing_data/legitimate_users.txt', sep='\t', quoting=csv.QUOTE_NONE,
                           names=["ID","Created_At","Collected_At","NumberOfFollowings", "NumberOfFollowers", "NumberOfTweets", "LengthOfScreenName", "LengthOfDescriptionInUserProfile","class"])
ham_profile_test['class'] = 0

spam_profile_test = pandas.read_csv('Testing_data/spammers.txt', sep='\t', quoting=csv.QUOTE_NONE,
                           names=["ID","Created_At","Collected_At","NumberOfFollowings", "NumberOfFollowers", "NumberOfTweets", "LengthOfScreenName", "LengthOfDescriptionInUserProfile","class"])

spam_profile_test['class'] = 1

result_profile_test = [ham_profile_test,spam_profile_test]
total_file_profile_test = pandas.concat(result_profile_test)
# print total_file.describe()
############################################################################################


#########################################################################################
##GETTING THE NUMBER OF FOLLOWERS FOR TESTING

ham_following_test = pandas.read_csv('Testing_data/legitimate_users_followings.txt', sep='\t', quoting=csv.QUOTE_NONE,
                           names=["ID","SeriesOfNumberOfFollowings","class"])
ham_following_test['class'] = 0

spam_following_test = pandas.read_csv('Testing_data/spammers_followings.txt', sep='\t', quoting=csv.QUOTE_NONE,
                           names=["ID","SeriesOfNumberOfFollowings","class"])

spam_following_test['class'] = 1

result_following_test = [ham_following_test,spam_following_test]
total_file_following_test = pandas.concat(result_following_test)
# print total_file_following_test['SeriesOfNumberOfFollowings'].head()
############################################################################################

################################################################################
## ADDIDITONAL COLUMNS FOR FEATURE EXTRACTION

total_file_following_test['Avg_follower_count'] = total_file_following_test.SeriesOfNumberOfFollowings.apply(avg_fol_count_test)
total_file_following_test['Var_follower_count'] = total_file_following_test.SeriesOfNumberOfFollowings.apply(var_fol_count_test)


####################################################################################################
######### COMBINATION OF FEATURES  ######################################################

avg_fol_count_test = total_file_following_test['Avg_follower_count']
var_fol_count_test = total_file_following_test['Var_follower_count']
length_profile_desc_test = total_file_profile_test['LengthOfDescriptionInUserProfile']

S1_test = pandas.Series(avg_fol_count_test)
S2_test = pandas.Series(var_fol_count_test)
S3_test = pandas.Series(length_profile_desc_test)
df1_test = pandas.DataFrame(S1_test)
df2_test = pandas.DataFrame(S2_test)
df3_test = pandas.DataFrame(S3_test)

result_combine_test = pandas.concat([df1_test,df2_test,df3_test],axis = 1)


################################################
##### USING CLASSIFIER TO TEST ################

print 'USING DECISION TREE'
#################### USING DECISION TREE TO PREDICT ########################################
predicted_labels = classifier_tree.predict(result_combine_test)

print("Confusion Matrix")
print(confusion_matrix(total_file_profile_test['class'], predicted_labels))
print("Precision")
print(precision_score(total_file_profile_test['class'], predicted_labels, average=None))
print("Recall")
print(recall_score(total_file_profile_test['class'], predicted_labels, average=None))
print("F1 score")
print(f1_score(total_file_profile_test['class'], predicted_labels, average=None))
print 'accuracy', accuracy_score(total_file_profile_test['class'], predicted_labels)

##############################################################################################

print 'USING ADA BOOST'
#################### USING ADA BOOST TO PREDICT ########################################
predicted_labels = classifier_ada.predict(result_combine_test)

print("Confusion Matrix")
print(confusion_matrix(total_file_profile_test['class'], predicted_labels))
print("Precision")
print(precision_score(total_file_profile_test['class'], predicted_labels, average=None))
print("Recall")
print(recall_score(total_file_profile_test['class'], predicted_labels, average=None))
print("F1 score")
print(f1_score(total_file_profile_test['class'], predicted_labels, average=None))
print 'accuracy', accuracy_score(total_file_profile_test['class'], predicted_labels)

##############################################################################################

print 'USING LOGISTIC REGRESSION'
#################### USING LOGISTIC REGRESSION TO PREDICT ########################################
predicted_labels = classifier_logistic.predict(result_combine_test)

print("Confusion Matrix")
print(confusion_matrix(total_file_profile_test['class'], predicted_labels))
print("Precision")
print(precision_score(total_file_profile_test['class'], predicted_labels, average=None))
print("Recall")
print(recall_score(total_file_profile_test['class'], predicted_labels, average=None))
print("F1 score")
print(f1_score(total_file_profile_test['class'], predicted_labels, average=None))
print 'accuracy', accuracy_score(total_file_profile_test['class'], predicted_labels)

##############################################################################################

print 'USING SUPPORT VECTOR MACHINE'
#################### USING SVM TO PREDICT ########################################
predicted_labels = classifier_svm.predict(result_combine_test)

print("Confusion Matrix")
print(confusion_matrix(total_file_profile_test['class'], predicted_labels))
print("Precision")
print(precision_score(total_file_profile_test['class'], predicted_labels, average=None))
print("Recall")
print(recall_score(total_file_profile_test['class'], predicted_labels, average=None))
print("F1 score")
print(f1_score(total_file_profile_test['class'], predicted_labels, average=None))
print 'accuracy', accuracy_score(total_file_profile_test['class'], predicted_labels)

##############################################################################################
