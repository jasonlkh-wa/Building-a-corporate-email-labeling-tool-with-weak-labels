# Building a corporate email labeling tool with weak labels (Snorkel used).
Everyday, we receive tones of emails in our corporate email address including job-related emails, internal announcements, external spam emails and etc. This issue becomes more significant when we look at the company's level. Therefore, before we start any machine learning projects to explore valuable insights, it is desirable to filter the unnecessary emails to reduce the noise in our dataset.

The objective of this machine learning project is to build an email labeling tool different types of emails by leveraging supervised and unsupervised machine learning techniques. The expected output of the models is to classify the emails as spam and non-spam (ham), and further label the ham emails into topics identified by the topic model.

The tool alone can be used in two ways:
1. Spam Classifier
2. Topic Model

Also, this tool is of benefit to other email machine learning project by means of:
1. Helping the users to select appropriate emails into the training data according to the labels generated.
2. Using the labels as extra features in other machine learning models (like using sentiment score as a new feature, topic scores are used instead)  

### Challenge of the project - Unlabeled Data
To build the spam classifier, labeled data are needed to validate the classifier regardless supervised or unsupervised method is used. However, the emails in the dataset are all **unlabeled** that I need to manually label **~8,000** emails before building any models. This is time-consuming and not practical in corporate settings with 2 reasons: 
1. Not every company is willing to spend many resources on the exploratory project. 
2. It is hard to maintain the model monitoring process after deployment (need to label large amount of emails before every assessment).

Therefore, I instead adopted a **weak-supervised** method to label the emails which dramatically reduce the number of manually labeled data up to 20% of the orginal amount. **Snorkel** - is the package I used in this project considering its flexibility to adopt different types of labeling functions.


## Project Flow
The final output of the model consists of multiple models and the development stages of the project are as below:
  1. Data Collection and Understanding the Data
  2. Generate Weak Labels with Snorkel Label Model
      - Testing different labeling functions
  3. Building Spam Classification Models
      - Exploring the difference between TF-IDF, Word2Vec and Prebuilt Word2Vec 
  4. Building the Topic Model
  5. Final Model Evaluation
   
![image](https://user-images.githubusercontent.com/50670119/153826702-e2fc293f-6533-431c-bcc7-2fe853fc51af.png)
_Project Development Stages_

## Machine Learning Techniques used in the project
  ### Weak Supervised Model:
  - Snorkel Label Model
  
  ### Word Embedding Techniques:
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
  - Latent Dirichlet allocation
  - Latent semantic analysis
 
## Dataset
#### Enron email dataset will be used used in this project. To replicate the email patterns in a department's mail box, I picked one employee's mail box for developing the models. (Assuming the employee worked for only one department during the period)

  - Enron email raw dataset: https://www.kaggle.com/wcukierski/enron-email-dataset

#### Jeff Dasovich's mail box will be used. There are 26,371 emails in total from 63 folders from 1999 to 2002.

