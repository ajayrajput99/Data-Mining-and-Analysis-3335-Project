#!/usr/bin/env python
# coding: utf-8

# # Fake News Detection using NLP Techniques and Machine Learning Model.

# ## Ajay Rajput (c0871742), Rahul Rawal (c0871230), Dhru Prajapati (c0867085)

# In[ ]:





# In[ ]:





# ## (1.) Importing Required Libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# ## (2.) Loading Dataset

# In[2]:


pd.set_option('display.max_colwidth', 100)

true_news_df = pd.read_csv('True.csv')
fake_news_df = pd.read_csv('Fake.csv')


# In[3]:


true_news_df.head(10)


# In[4]:


fake_news_df.head(10)


# In[ ]:





# ### (i.) Shapes of both datasets

# In[5]:


print("Fake News: ",fake_news_df.shape)
print("True News: ",true_news_df.shape)


# In[ ]:





# ### (ii.) Exploring Datatypes and Data Description

# In[6]:


## Data Types and Null Values...

fake_news_df.info()


# In[7]:


## Data Types and Null Values...

true_news_df.info()


# In[ ]:





# ### (iii.) Fake News Description

# In[8]:


fake_news_df.describe()


# In[ ]:





# ### (iv.) True News Description

# In[9]:


true_news_df.describe()


# In[ ]:





# ### (v.) Unique Subjects 

# In[10]:


print("Fake News Subjects:", fake_news_df['subject'].unique())
print("True News Subjects:", true_news_df['subject'].unique())


# In[ ]:





# ### (vi.) Dropping Null Values

# From below description it is clear that we do not have any null values in the dataset, meaning that feature in each row contains
# some values.

# In[11]:


fake_news_df.isnull().sum()


# In[12]:


true_news_df.isnull().sum()


# ### (vii.) Droping Duplicate Values

# In[13]:


fake_news_df.drop_duplicates()


# In[14]:


duplicates = true_news_df.drop_duplicates()

duplicates


# ## (3.)  Adding Lables to Fake_News and True_News Datasets

# In[15]:


fake_news_df['label'] = 0
true_news_df['label'] = 1


# In[16]:


fake_news_df.head()


# In[17]:


true_news_df.head()


# In[ ]:





# ## (4.) Concating both the Data-Framses to make a single Dataset (Final_News)

# In[18]:


final_news_df = pd.concat([fake_news_df, true_news_df], axis = 0)


# In[19]:


final_news_df = final_news_df.sample(frac= 1)


# In[20]:


final_news_df.reset_index(inplace= True)


# In[21]:


final_news_df.drop(final_news_df.columns[0], axis=1, inplace= True)


# In[22]:


final_news_df


# In[ ]:





# ## (5.) Feature Engineering

# In[23]:


import string
from nltk.tokenize import word_tokenize
import re
import nltk
from nltk.corpus import stopwords


# ### (i.) Punctuation_count

# string.punctuation
# 
# '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

# In[24]:


def punct_count(text):
    
    text_length = sum([1 for x in text if x in string.punctuation])
    
    return text_length


# In[25]:


final_news_df['punct_count'] = final_news_df['text'].apply(lambda x : punct_count(x))


# In[ ]:





# ### (ii.) text_body_length

# In[26]:


final_news_df['text_body_length'] = final_news_df['text'].apply(lambda x: len(x) - x.count(" "))


# In[ ]:





# In[27]:


final_news_df.head(10)


# In[ ]:





# In[ ]:





# ### (iii.) Droping Unwanted Data Fields(Features)

# As the "Date" field and the "Subject" field does not contribute much into our dataset and machine learning model..!

# In[28]:


final_news_df.drop(['subject', 'date'], axis=1, inplace= True)


# In[ ]:





# In[ ]:





# ## (6.) Data Prepration for Machine Learning Model  (Wrangaling and Cleaning )

# In[29]:


## Importing Vectorizer and Initializing Lemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer

lemmatizer = nltk .WordNetLemmatizer()


# ### (i.) Dropping Data points that has zero(0) value in text_body_lenght 

# In[ ]:





# In[30]:


zero_value_indexs = final_news_df[final_news_df['text_body_length'] == 0].index

final_news_df.drop(index= zero_value_indexs, inplace= True)


# In[ ]:





# 

# ### (ii.) Cleaning data (Removing punctuation, stopwords and Lemmatizing the text)

# In[31]:


def data_cleaning(text):
    
    ## converting text into lower case
    data = text.lower()
    
    ## Removing Punctuations
    no_punct_data = re.sub(r'\W+', ' ', data)
    
    ## tokenizing the text
    tokenize_text = re.split(r'\s', no_punct_data)
    
    ## Stopwords
    stop_words = stopwords.words('english')
    
    ## Removing Stopwords and Lemmatizing the text
    clean_data = " ".join([lemmatizer.lemmatize(x) for x in tokenize_text if x not in stop_words])
    
    
    return clean_data


# In[ ]:





# In[32]:


final_news_df['text'] = final_news_df['text'].apply(lambda x: data_cleaning(x))


# In[33]:


final_news_df['title'] = final_news_df['title'].apply(lambda x: data_cleaning(x))


# In[34]:


# final_news_df.head()


# In[ ]:





# ### (iii.) Adding tokenize list of text_body into Dataset

# In[35]:


final_news_df['tokenized_text'] = final_news_df['text'].apply(lambda x: word_tokenize(x))


# In[36]:


final_news_df.head()


# In[ ]:





# ## (7.) Visualizing some key factors of the dataset

# In[ ]:





# ### (i.) Historgam for Text Body lengthh 

# In[37]:


import seaborn as sns
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt


# wordcloud
from wordcloud import WordCloud


# In[38]:


fake_data = final_news_df[final_news_df['label'] == 0]
true_data = final_news_df[final_news_df['label'] == 1]


# In[39]:


# create histogram of text body lengths for fake news articles
plt.hist(fake_data['text_body_length'], bins=10, alpha=0.5, label='Fake', )

# create histogram of text body lengths for true news articles
plt.hist(true_data['text_body_length'], bins=10, alpha=0.5, label='True')


plt.xlabel('Text Body Length')

plt.ylabel('Count')

plt.legend(loc='upper right')


## Saving the plot 

plt.savefig("Hist1.png")


plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# ### (ii.) Word Cloud of Textbody Tokens

# ### (a.) Fake News Tokens WordCloud

# In[40]:


fake_news_tokens = final_news_df[final_news_df['label'] == 0]['tokenized_text']


# In[41]:


fake_news_text =  str([" ".join(x) for x in fake_news_tokens])


# In[ ]:


fake_wordcloud = WordCloud(width= 800, height= 400, max_words= 100, collocations= True).generate(fake_news_text)


# In[ ]:


plt.figure(figsize=(12, 8))
plt.imshow(fake_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.savefig("FakeWordCloud.png")
plt.show()


# In[ ]:





# ### (b.) True News Tokens WordCloud

# In[ ]:


true_news_tokens = final_news_df[final_news_df['label'] == 1]['tokenized_text']


# In[ ]:


true_news_text =  str([" ".join(x) for x in true_news_tokens])


# In[ ]:


true_wordcloud = WordCloud(width= 800, height= 400, max_words= 100, collocations= True).generate(true_news_text)


# In[ ]:


plt.figure(figsize=(12, 8))
plt.imshow(true_wordcloud, interpolation='bilinear')
plt.title = ""
plt.axis('off')
plt.show()


# In[ ]:





# In[ ]:





# ## (8.) Spliting Data into Train and Test Sets

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = final_news_df['text']
Y = final_news_df['label']


# In[ ]:


#Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[ ]:





# #### (i.) Shape of Train and Test data

# In[ ]:


print(X_train.shape , X_test.shape)


# In[ ]:





# ## (9.) Text Vectorization

# In[ ]:


## We are using TFIDF Vectorizer.

from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


vectorizer = TfidfVectorizer(max_features=50000, lowercase= False, ngram_range=(1,2))


# In[ ]:


X_train_vect = vectorizer.fit_transform(X_train)

X_train_vect = X_train_vect.toarray()


# In[ ]:


X_test_vect = vectorizer.transform(X_test)

X_test_vect = X_test_vect.toarray()


# In[ ]:





# ### (i.) Shape of vectorized Train and Test data

# In[ ]:


print(X_train_vect.shape, X_test_vect.shape)


# In[ ]:





# ### (ii.) Saving Vectorized Data into Datasets

# In[ ]:


training_data = pd.DataFrame(X_train_vect , columns=vectorizer.get_feature_names_out())
testing_data = pd.DataFrame(X_test_vect , columns= vectorizer.get_feature_names_out())


# In[ ]:


training_data.head(5)


# In[ ]:





# ## (10.) Initializign Model and Training Model

# In[ ]:





# In[ ]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# In[ ]:


# Train a Multinomial Naive Bayes model on the training data

model = MultinomialNB()

model.fit(training_data, y_train)


# In[ ]:





# In[ ]:


# Make predictions on the Testing data using the trained model
y_pred = model.predict(testing_data)

# Evaluate the accuracy of the model
testdata_accuracy = accuracy_score(y_test, y_pred)

print("Test set Accuracy:", testdata_accuracy * 100)


# In[ ]:





# In[ ]:


# Make predictions on the Training data using the trained model
y_train_pred = model.predict(training_data)

# Evaluate the accuracy of the model
traindata_accuracy = accuracy_score(y_train, y_train_pred)

print("Training set Accuracy:", traindata_accuracy)


# In[ ]:





# In[ ]:


model.score(training_data, y_train)


# In[ ]:


model.score(testing_data, y_test)


# In[ ]:





# ## (11.) Saving Model

# In[ ]:


import joblib

joblib.dump(model, 'model.pkl')


# In[ ]:


new_data = data_clean(str("ChatGPT is generating fake news stories — attributed to real journalists. I set out to separate fact from fiction."))


# In[ ]:


model = joblib.load('model.pkl')


new_vect_data = vectorizer.transform([new_data]).toarray()

# df = pd.DataFrame(new_vect_data, vectorizer.get_feature_names_out)

predicted_labels = model.predict(new_vect_data)

print(predicted_labels)

if predicted_labels[0] == 0:
    
    print("This is a Fake News")

else:

    print("This is a True News")
    


# In[ ]:




