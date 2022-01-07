# Using machine learning to explore corporate email data
#### Everyday, we receive tones of emails in our corporate email address including job-related emails, internal announcements, external spam emails and etc. This issue becomes more significant when we look at the company's level. Therefore, before we start any machine learning projects to explore valuable insights, we need to filter out the unnecessary emails to reduce the noise in our dataset.

#### The goal of this machine learning project is to build a tool for classifying different types of emails by leveraging supervised, unsupervised and semi-supervised machine learning techniques. The expected outcome of the models is to classify the emails as spam and non-spam (ham), and further label the ham emails into categories found by unsupervised learning algorithm. 

#### The tool is of benefit to other email machine learning project by means of:
1. Helping the users to pick appropriate emails into the training data according to the labels generated.
2. Using the labels as one of the features in their machine learning model (similar to using sentiment score as a new feature).


## Project Flow
#### The final output of the model consists of multiple models and the development stages of the project are as below:
  1. Data Collection and Understanding the Data
  2. Building Spam Classification Models
      - Data Cleaning and Preprocessing
      - Exploring the difference between Bag of Words (BoW), TF-IDF, Word2Vec and Prebuilt Word2Vec
      - Building Basic Supervised Models 
      - Model Tuning 
      - Building Ensemble Models
      - Final Evaluation and Model Selection
  3. Topic Modelling
      - Data Cleaning and Preprocessing
      - Building Unsupervised Models
      - Evaluating the performance of the models with Silhouette Score and Fowlkes Mallows Score
      - Exploring the labels generated
  4. Future planning on improving the models and Use Case Exploration  
![image](https://user-images.githubusercontent.com/50670119/148485472-8a7e6315-e264-4be6-ac53-0866fbb3bdb6.png) 
_Project Development Stages_

## Machine Learning Techniques used in the project
  ### Word Embedding Techniques:
  - Bag of Words (BoW)
  - TF-IDF
  - Word2Vec
  - Prebuilt Word2Vec from Google
 
  ### Supervised Learning Models:
  - Logistic Regression
  - Support Vector Machine
  - Decision Tree
  - Random Forest 
  - XGBoost
  
  ### Unsupervised Learning Models:
  - KMean
  - Latent Dirichlet allocation
  - Latent semantic analysis

## Data Collection and Understanding the Data
#### Enron email dataset and Hillary Clinton email dataset are used in this project.
  - Enron email raw dataset: https://www.kaggle.com/wcukierski/enron-email-dataset
  - Labeled Enron email dataset (ham/spam): http://nlp.cs.aueb.gr/software_and_datasets/Enron-Spam/index.html
  - Hillary Clinton email: https://www.kaggle.com/kaggle/hillary-clinton-emails  
  
#### The Enron email dataset will be the main dataset used throughout the project while the Hillary Clinton email will be used for validating the models built and model methodologies.

#### As the Hillary Clinton email dataset behaves more close to personal emails while Enron's emails are more related to the business side. It is expected that the ML models built based on Enron emails cannot be directly applied to Hillary Clinton email dataset. Hence, this project will also study if the model methodologies is transferable to other email data. 

