#!/usr/bin/env python
# coding: utf-8

# 
# Problem Statement
# 
# Credit score cards are used in the financial industry to predict the likelihood of credit card defaults and guide the issuance of credit cards. These scores rely on applicants' personal data and historical information, quantifying risk objectively.

# This project wants to create Random Forest Classifier Model to help banks decide who should get a credit card

# In[1]:


# loding the essential library

# general libraries
import pandas as pd
import numpy as np

# visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# loading the datatset containing application record
df_application_record= pd.read_csv('application_record.csv')

df_application_record.head(5)


# # Column Description of Application Record Dataset:
# 
# Feature Name         	Explanation 	Remarks
# ID 	Client number 	
# CODE_GENDER 	        Gender 	
# FLAG_OWN_CAR 	        Is there a car 	
# FLAG_OWN_REALTY     	Is there a property 	
# CNT_CHILDREN 	        Number of children 	
# AMT_INCOME_TOTAL 	    Annual income 	
# NAME_INCOME_TYPE 	    Income category 	
# NAME_EDUCATION_TYPE 	Education level 	
# NAME_FAMILY_STATUS   	Marital status 	
# NAME_HOUSING_TYPE 	    Way of living (House Type) 	
# DAYS_BIRTH 	               Birthday 	              Count backwards from current day (0), -1 means yesterday
# DAYS_EMPLOYED 	         Start date of employment     Count backwards from current day(0).If positive, it means the person                                                           currently unemployed.
# FLAG_MOBIL 	           Is there a mobile phone 	
# FLAG_WORK_PHONE 	   Is there a work phone 	
# FLAG_PHONE 	           Is there a phone 	
# FLAG_EMAIL 	           Is there an email 	
# OCCUPATION_TYPE 	   Occupation 	
# CNT_FAM_MEMBERS 	   Family size

# # Loading Dataset(Application Record)
# 

# In[3]:


# checking shape of the dataset (application_record)

print('The number of rows in Application Record dataset is', df_application_record.shape[0] , '\n')

print('The number of columns in Application Record dataset is', df_application_record.shape[1])


# # Getting know about the dataset(Application Record)

# In[4]:


# checking datatypes
df_application_record.dtypes


# # Observation
# 
#     The dataset of application record contains 438557 rows and 18 columns
#     This dataset contains eight of columns of object datatype
#     And remaining 10 columns has of numric datatype
# 

# # Getting know about the dataset(Credit Record)

# In[5]:


#loading the datatset containing credit record

df_credit_record = pd.read_csv('credit_record.csv')


df_credit_record.head()


# In[6]:


#checking shape of the dataset (credit_record)

print('The number of rows in Credit Record dataset is', df_credit_record.shape[0] , '\n')

print('The number of columns in Credit Record dataset is', df_credit_record.shape[1])


# In[7]:


# checking data types

df_credit_record.dtypes


# # Observation
# 
#     The dataset of credit record contains 1048575 rows and 3 columns
#     This dataset contains 1 columns of object datatype which shows Status of billing details status of creddit card
#     And remaining 2 columns has numric datatype
# 

# # Merging two datasets (Basis of common ID)

# In[8]:



df = pd.merge(df_application_record, df_credit_record, on='ID' , how='inner')


# -We merge two dataframes on basis of ID 
# column present in both datasets.
# -Moreover, Credit record dataset has more records of clients than Application record dataset, so we make sure that our new      datafrmae df contains only those rows which have the same ID number
# -For that purose we use inner parameter in pd.merge function

# # Previewing the merged dataset

# In[9]:


df.head()


# # Getting know about the merged dataset

# Observation:
# 
#     At earlier we have seen that Application record dataset has 438557 rows
#     Now, our merged dataset has only contain 777715
#     That means Application Record & Credit Record dataset has only 777715 rows which have common ID numbers
# 

# # Checking for missing values & Dealing with them

# In[10]:


print(f'The number of rows in merged dataset (df) = {df.shape[0]} \n The number of columns in merged dataset (df) is = {df.shape[1]}' )


# 
# Observation:
# 
#     At earlier we have seen that Application record dataset has 438557 rows
#     Now, our merged dataset has only contain 777715
#     That means Application Record & Credit Record dataset has only 777715 rows which have common ID numbers
# 

# In[11]:


# checking for null values

df.isna().sum() 


# In[12]:


# plotting the heatmap to check for null values

# definig the figure size
plt.figure(figsize=(10, 6))

# plotting the heatmap of missing values
sns.heatmap(df.isna(), cbar=False , yticklabels=False, cmap='viridis')

# defining the title
plt.title('Heatmap for Missing Values')
plt.xticks(rotation=50)
plt.show()


# In[13]:


# check for the percentage of missing values in `OCCUPATION_TYPE` column

df['OCCUPATION_TYPE'].isna().sum() / df.shape[0] * 100


# # Observation
# 
#     There are only one column in our datafrme which has null values.
# 
#     This column is OCCUPATION_TYPE and it has 240048 missing values and it has percentage of 30.86% missing values
# 
#     We have only one option which is to drop the column OCCUPATION_TYPE because it has a high percentage of missing values and secondly every person has itd own unique record. and we cannot simply impute the missing values with modbservation:
# 
#     There are only one column in our datafrme which has null values.
# 
#     This column is OCCUPATION_TYPE and it has 240048 missing values and it has percentage of 30.86% missing values
# 
#     We have only one option which is to drop the column OCCUPATION_TYPE because it has a high percentage of missing values and secondly every person has itd own unique record. and we cannot simply impute the missing values with mod

# In[14]:


# dropping the column `OCCUPATION_TYPE`
df.drop('OCCUPATION_TYPE', axis=1, inplace=True) 


# In[15]:


# confirming the outcome 
df.columns


# # Checking for unique values count in dataframe

# In[16]:


# check for the count of unique values in each column
df.nunique()


# # Observation:
# 
#     Our dataset has 777715 rows but there are only 36457 unique values in ID column.
#     This shows that there maybe duplicates in our dataset
# 

# # Checking for duplicates

# In[17]:


df.duplicated().sum()


# In[18]:


df[df['ID'].duplicated()].head(10)


# In[19]:


df[df['ID'].duplicated()].tail(20)


# # Observation:
#     
# 
#     By carefully checking the dataset we can say that on basis of number of unique enteries in ID column we have data of 36457 clients.
#     And there are no duplicates in our dataset the data of 36457 ID which is collected on the basis of different months of MONTHS_BALANCE
# 

# # Renaming the columns

# In[20]:



df.rename(columns={
    'CODE_GENDER': 'gender',
    'FLAG_OWN_CAR': 'own_car',
    'FLAG_OWN_REALTY': 'own_property',
    'CNT_CHILDREN': 'children',
    'AMT_INCOME_TOTAL': 'income',
    'NAME_INCOME_TYPE': 'income_type',
    'NAME_EDUCATION_TYPE': 'education',
    'NAME_FAMILY_STATUS': 'family_status',
    'NAME_HOUSING_TYPE': 'housing_type',
    'FLAG_MOBIL': 'mobile',
    'FLAG_WORK_PHONE': 'work_phone',
    'FLAG_PHONE': 'phone',
    'FLAG_EMAIL': 'email',
    'CNT_FAM_MEMBERS': 'family_members',
    'MONTHS_BALANCE': 'months_balance',
    'STATUS' : 'status',
    'DAYS_BIRTH' : 'age_in_days',
    'DAYS_EMPLOYED' : 'employment_in_days'

} , inplace=True)


# In[21]:


df.columns


# # Mapping the values in a meaningful way

# In[22]:


df.select_dtypes(include='object').columns


# In[23]:


col = ['gender', 'own_car', 'own_property']

for i in col:
    print(f'{df[i].value_counts()}')


# In[24]:


# maping the values in some columns


# mapping unique enteries of gender
df['gender'] = df['gender'].map({'F':'female', 'M': 'male'})

# mapping unique enteries of own_car
df['own_car'] = df['own_car'].map({'N': 'no', 'Y': 'yes'})

# mapping unique enteries of own_property
df['own_property'] = df['own_property'].map({'N': 'no', 'Y': 'yes'})


# In[25]:


# check for unique values in status column

df['status'].value_counts()


# # Feature Engineering

# In[26]:


# maping the values in status column and storing result in new column

df['loan_status'] = df['status'].map({'0': 'first_month_due' , '1': '2nd_month_due', '2' : '3rd_month_overdue',  '3': '4th_month_overdue',
                                '4' : '5th_month_overdue', '5' : 'bad_debt' , 'C': 'good' , 'X' : 'no_loan'})


# In[27]:


# confirming the outcome

df.columns.values


# # Exploratory Data Analysis

# # Purpose:
# 
#     The main goal of the analysis is to get ideas about the different attributes of the clients.
#     To get an overview of the distribution of the data.
#     To get an overview of the relationship between the attributes.
#     Particularly, to get an overview of the relationship between the attributes and the target variable[loan_status].
# 

# # Checking for distribution of [gender , own_car,  own_property , income_type]

# In[28]:


# Define the list of column names
columns = ['gender', 'own_car', 'own_property']

# Create subplots for each column
plt.figure(figsize=(16 , 9))  # Adjust the figure size as needed

for i in range(len(columns)):
    plt.subplot(1, 3, i+1)
    plt.title(columns[i])  # Use the column name as the title
    
    # Plot pie chart
    counts = df[columns[i]].value_counts()
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%'  )
    
    # Add legend with unique values
    plt.legend(counts.index)
plt.show()


# In[29]:


# creating list of specific columns
col = ['gender', 'own_car', 'own_property', 'income_type',]

# defining the figure size
plt.figure(figsize=(15, 6))

# plotting the countplot using for loop
for i in range(len(col)):
    # defining the subplot
    plt.subplot(2, 2, i+1)
    # adding title
    plt.title(col[i])
    # plotting the countplot
    sns.countplot(data=df, x=df[col[i]])
    # rotating the x-axis labels
    plt.xticks(rotation=45)
# layout adjustment
plt.tight_layout()
plt.show()


# # Observation:
# 
#     There are more Female clients in our dataframe than the male clients
#     The number of clients who don't have car is more than the number of clients who have car
#     More number of clients have their own property
# 

# In[30]:


# checking the count of loan status

df['loan_status'].value_counts()


# # Observation:
# 
#     Most of the clients in our dataset have paid off their loan for that month
#     first month loan due and no loan for the month comes at 2nd and 3rd place respectively in terms of value counts
#     1527 clients have bad debt
# 

# # Getting insight from loan status v/s gender

# plotting count plot of loan status v/s gender

# In[31]:


# checking the relationship between loan status and gender

# defining the figure size
plt.figure(figsize=(15, 5))

# plotting the countplot
sns.countplot(data=df, x=df['loan_status'], hue=df['gender'])

# defining the tilte
plt.title('loan status v/s gender')
# rotating the x-axis labels
plt.xticks(rotation=25)
plt.show()


# # getting the value count of loan status v/s gender which are less than 10000

# In[32]:


# value count of loan status v/s gender
filtered_loan = df.groupby('loan_status')['gender'].value_counts()

# filtering the values less than 10000
filtered_loan[filtered_loan<10000].unstack()


# Bar Plot of the relationship between loan_status and gender which have values less than 10000

# In[33]:


# value count of loan status v/s gender
filtered_loan = df.groupby('loan_status')['gender'].value_counts()

# filtering the values less than 10000

plt.figure(figsize=(15, 9))

filtered_loan[filtered_loan<10000].unstack().plot(kind='barh')

plt.show()


# # Observation:
# 
#     As our dataset has more number of females than males so the count of female is larger than the count of males who have paid off their loan for that month and have no loan for the month
#     Moreover, the females have more over due and bad debt than males
# 

# In[34]:


df['income_type'].value_counts()


# 
# Observation:
# 
#     As our dataset has more number of females than males so the count of female is larger than the count of males who have paid off their loan for that month and have no loan for the month
#     Moreover, the females have more over due and bad debt than males
# 

# # Getting insight from loan status v/s gender

# In[35]:


count = df['income_type'].value_counts()

plt.figure(figsize=(15, 6))

# Plotting the countplot for each unique value of 'income_type'
for i in range(len(count)):
    plt.subplot(2, 3, i+1)
    plt.title(count.index[i])  # Use the unique value of 'income_type' as the title
    sns.countplot(data=df[df['income_type'] == count.index[i]], x='gender')

plt.tight_layout()  # Adjust the layout to prevent overlapping
plt.show()


# # Observation
# 
#     In Working and Commercial assciate income_type the number of male clients is half of the female clients
#     Whereas, in income_type = Pensioner and  Students males ATM clients numbers are very less as compared to females
# 

# In[36]:


# checking the relationship between loan status and income type

# defining the figure size
plt.figure(figsize=(15, 5))

# plotting the countplot
sns.countplot(data=df, x=df['loan_status'], hue=df['income_type'])

# defining the tilte
plt.title('Count plot of loan status v/s income type')
plt.xticks(rotation=25)
plt.show()


# # Observation:
# 
#     Most of the clients with working income_type have paid off their loan for that month
#     Similar trend can be seen in the income catagories like first_month_due & no_loan_for_month working catagory comes top of the list followed by commercial associates & pensioners
# 

# # getting insights from gender v/s income type & loan status with value counts less than 1000

# In[37]:




count = df.groupby('gender')[['income_type' , 'loan_status']].value_counts()

count[count<1000].unstack()


# In[38]:


# checking the relationship between gender v/s income type & loan status 
count = df.groupby('gender')[['income_type', 'loan_status' ]].value_counts()

# plotting the barh plot for [gender v/s income type & loan status] value count less than 1000
count[count<1000].unstack().plot(kind='bar' , figsize=(15, 5) , legend=True ) 
# defining the tilte
plt.title('Relationship between gender v/s loan status and income type [value count < 1000]')
# rotating the x-axis labels
plt.xticks(rotation=25)
plt.show()


# Area Plot of the relationship between gender v/s income type & loan status with value counts less than 1000

#  checking the relationship between gender vs income type & loan status
# count = df.groupby('gender')[['loan_status','income_type']].value_counts()
# plotting the area plot for [gender vs income type & loan status] value count less than 1000
# count[count<1000].unstack().plot(kind='area' , figsize=(15, 9)) 
#  defining the tilte
# plt.title('Relationship between gender vs income type & loan status [value count < 1000]')
# rotating the x-axis labels
# plt.xticks(rotation=25)
# plt.show()

# # Observation:
# 
#     When we see loan_status trends in terms of income_type and gender we can see that the male with student inocme_type has no_loan for that month.
#     Also, In good & first month due loan_status the number male students is very low and compared to females.
#     State servants income type in both male and female gender have very less numbers in billing overdue of loan_status
#     Both males and females students has 1 and 0 numbers respectively in 2nd_month_due of loan_status
# 

# 
# Getting insights from education and gender
# checking the value count of education

# In[39]:


df.education.value_counts()


# # value count of education vs gender
# 

# In[40]:


# checking the relationship between gender and education
df.groupby('gender')[['education']].value_counts()


# # plotting the value count of education vs gender
# 

# # Observation:
# 
#     As our dataset has larget number of people with secondary education.
#     Therefore, the number of male and female with secondary education is high.
#     Moreover, the number of Lower_secondary & Academic degree is very less in both male and female
#     The proportion of Lower_secondary eduaction is similar for both genders
# 

# 
# Getting insights from education and loan_status

# In[41]:


# checking the relationship between loan_status and education using groupby function
df.groupby('education')[['loan_status']].value_counts().unstack().plot(kind='bar', figsize=(15, 5) , stacked= True)
# defining the tilte
plt.title('Relationship between loan_status and education')
# rotating the x-axis labels
plt.xticks(rotation=25)
plt.show()


# plotting the barh plot for education vs loan_status [values < 500]

# In[42]:


# checking the relationship between loan_status and education
count = df.groupby('education')[['loan_status']].value_counts()

# plotting the barh plot for education vs loan_status which have values less than 500
count[count<=500].unstack().plot(kind='barh' , figsize=(15, 6))
# defining the tilte
plt.title('Relationship between loan_status and education [value count < 500]')
# rotating the x-axis labels
plt.xticks(rotation=25)
plt.show()


# Observation:
# 
#     As our dataset has larget number of people with secondary education and they have largest numbers who have paid off their loan
#     A similar trend can be seen in the Academic degree education
#     The mostly people who have bad debt are from Lower_secondary & Incomplete higher education
#     
# 
#     
# Getting insights from housing_type and loan_status
# 
# 
# unique enteries & value count from housing_type

# In[43]:


df['housing_type'].unique()


# In[44]:


# checking value counts of `housing_type` column
df['housing_type'].value_counts()


# In[45]:


# checking the relationship between loan_status and housing_type

# defining the figure size
plt.figure(figsize=(15, 5))

# plotting the countplot
sns.countplot(data=df, x=df['loan_status'], hue=df['housing_type'])

plt.xticks(rotation=25)
plt.title('Relationship between loan_status and housing_type')

plt.show()


# 
# filtering the value counts housing & loan_status==bad_debt¶
# 

# In[46]:


# checking the relationship between loan_status and housing_type

# Group by 'housing_type' and 'loan_status' to get their counts
count = df.groupby(['housing_type', 'loan_status']).size()

# Get value counts where loan_status is 'bad_devit'
bad_loan= count[count.index.get_level_values('loan_status') == 'bad_debt']
bad_loan


# # plotting the value count < 1000 from housing_type & loan_status

# In[47]:


count = df.groupby(['housing_type', 'loan_status']).size()
count[count<1000].unstack().plot(kind='barh' , figsize=(15, 5) , legend=True )

plt.title('Relationship between loan_status and housing_type [value count < 1000]')

plt.show()


# # Observation:
# 
#     Mostly people who have their own house have good , first month due & no_loan loan_status which is a positive trend.
#     People who live with parents have largest share in 2nd_month_due loan_status. This trend is followed by Municipal apartment , Rented apartment and With office apartment respectively
#     The people with lowest number of bad_debt loan_status are from office apartment & co apartment housing type
# 

#      

# Getting insights from family_members and loan_status

# In[48]:


df.columns


# In[49]:


# checking value counts of `family_members`
df.family_members.value_counts()


# In[50]:


# checking the relationship between loan_status and family_members
df.groupby('family_members')[['loan_status']].value_counts().unstack()


# In[51]:


df.select_dtypes(exclude='object').columns # checking the column names with numeric datatype


# In[52]:


# checking value counts of `children`
df.children.value_counts()


# In[53]:


# checking the relationship between loan_status and children count

df.groupby('children')[['loan_status']].value_counts().unstack()


# # Getting insights from employment_in_days
# 
# 
# count of persons who are unemployed

# In[54]:


# plotting histogram of employment_in_days
df['employment_in_days'].plot(kind = 'hist')
# defining the title
plt.title('Frequency of employment_in_days')
plt.show()


# In[55]:


# Filter DataFrame where employment_in_days > 0 to show unemployment count
df[df['employment_in_days'] > 0].value_counts().sum()


# # Observations:
# 
#     The value in employment in days which are greater than 0 shows the status of the person is unemployed
#     We have 127972 persone who are unemployed
# 

#   

# # Checking the relationship between employment_in_days > 0 (unemployment) & loan_status

# In[56]:


# Filter DataFrame where employment_in_days is greater than 0
filtered_df = df[df['employment_in_days'] > 0]

# Group by loan_status and calculate value counts
filtered_df.groupby('loan_status').size()


# # Observations
# 
#     The loan_status of the most of the unemployed persons is good , first month due & no_loan which is a positive trend.
#     Whereas, the loan_status of the bad_debt for unemployed persons is very low.
# 

#   

# # Data Preprocessing

# In[57]:


df.columns # printing exact column names


# plotting the scatter plot of numeric columns

# In[58]:


# checking the relationship between loan_status and income
col = ['children', 'income', 'age_in_days',  'family_members' , 'employment_in_days']

# defining the figure size
plt.figure(figsize=(15, 6))

# plotting the countplot using for loop
for i in range(len(col)):
    plt.subplot(3, 2, i+1)
    plt.title(col[i])  # Use the column name as the title
    sns.scatterplot(data=df, y=col[i], x='ID')
plt.title('scatterplot of ID vs numeric columns')
plt.tight_layout()  # Adjust the layout to prevent overlapping
plt.show()


# In[59]:


df.columns


# plotting the boxplot of numeric columns

# In[60]:


# checking the relationship between loan_status and income

# filtering the list of specific columns    
col = ['children', 'income', 'age_in_days',  'family_members' ]

# defining the figure size
plt.figure(figsize=(15, 6))

# plotting the countplot using for loop
for i in range(len(col)):
    plt.subplot(3, 2, i+1)
    plt.title(col[i])  # Use the column name as the title
    sns.boxplot(data=df, y=col[i])

plt.tight_layout()  # Adjust the layout to prevent overlapping
plt.show()


# # Observations
# 
#     As we can see that the outliers are present in three columns:
#         income
#         children
#         family_members
#     Both scatter plot and box plot are showing outliers in the above mention columns
# 

#     

# Removing outliers from ['children', 'income' , 'family_members']

# In[61]:


# removing outliers

# filtering the list of specific columns
col = ['children', 'income' , 'family_members']

# for loop to remove outliers
for i in range(len(col)):
    # calculating the first and third quartile
    q1 = df[col[i]].quantile(0.25)
    q3 = df[col[i]].quantile(0.75)
    # calculating the interquartile range
    iqr = q3 - q1
    # calculating the lower and upper bounds
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    # replacing the outliers with the median
    df[col[i]] = np.where((df[col[i]] >= upper_bound) | (df[col[i]] <= lower_bound), df[col[i]].median() , df[col[i]])


# 
# again plotting boxplot to confirm outcomes

# In[62]:


col = ['children', 'income', 'age_in_days',  'family_members']

plt.figure(figsize=(15, 6))

for i in range(len(col)):
    plt.subplot(2, 2, i+1)
    plt.title(col[i])  # Use the column name as the title
    sns.boxplot(data=df, y=col[i])

plt.tight_layout()  # Adjust the layout to prevent overlapping
plt.show()


# In[63]:


df.select_dtypes(exclude='object').columns


# In[64]:


# stats libraries
import scipy.stats as stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

# machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score , classification_report, f1_score, precision_score, recall_score


# Standardization

# In[65]:


# filtering the list of specific columns
col = ['children', 'income', 'age_in_days',  'family_members']

# calling the standard scaler
sc = StandardScaler()
# for loop to scale the specific columns
for i in col:
    df[i] = sc.fit_transform(df[[i]])


# # Suitability of Model(Random Forest Classifier)

# In[66]:


# checking the distribution of loan_status
df.loan_status.value_counts().plot(kind='barh', figsize=(15, 5))
# defining the title
plt.title('Classs Distribution inside Loan Status')
plt.show()


# # Observation:
# 
#     The loan_status is our target variable.
#     If we see the class distribution, of our target variable loan_status, we can see that our target variable is highly imbalanced.
#     Which means that we cannot use logistic regression on our data.
#     That is why we need to use Random Forest Classifier for our model.
# 

# In[67]:


df.head()


# # Label Encoding

# In[68]:


df.columns # printing exact column names


# In[69]:


# filtering the list of specific columns which we need to encode
col = ['gender', 'own_car', 'own_property', 'income_type','education', 'family_status', 'housing_type', 'status']

# calling the label encoder
le = LabelEncoder()

# for loop to encode the specific columns
for i in col:
    df[i] =le.fit_transform(df[i] )


# In[70]:


df.head() # previewing the dataset


# # Best Features Selection: Random Forest Classifier
# Independence of Observations:
# 
# Correlation Matrix for Numerical Features
# 

# In[71]:


# correlation matrix

# defining the figure size
plt.figure(figsize=(15, 6))
# plotting the heatmap
sns.heatmap(df[['children', 'income', 'age_in_days',  'family_members', 'employment_in_days', 'months_balance']].corr(), annot=True , cbar=False) 
# defining the title and rotation of x-axis labels
plt.xticks(rotation=25)
plt.title('Heatmap for Correlation Matrix for Numerical Features')
plt.show()


# # Observation
# 
#     As we can see that children and family_members are highly correlated with each other
# 
#     But there is no strong correlation between other numerical features
# 
#     Hence, the condition of Independence of Observations is almost satisfied for Random Forest Classifier.
# 

#     

# # Check for Multicollinearity

# 
# Variance Inflation Factor

# In[72]:


# Drop 'loan_status' and 'status' columns from col
col = df.drop(['loan_status', 'status'], axis=1) # Assume that col is our independent variable

# Compute variance inflation factor

# Create a dataframe to store the VIF
factor  = pd.DataFrame(columns=["VIF", "Features"] )
# For each column, compute the VIF
factor["Features"] = col.columns
factor["VIF"] = [variance_inflation_factor(col.values, i) for i in range(col.shape[1])]

# Display the results
factor


# # Observation
# 
#     Multicollinearity occurs when two or more predictor variables in the model are highly correlated.
# 
#     Variance Inflation Factor (VIF) is 1.0 for all columns except children, family_status and mobile which are highly correlated with each other.
#     Also, the columns children and family_status are highly correlated with each other.
#     Hence, we need to drop children , family_status & mobile columns from our independent variable list to get best results from Random Forest Classifier.
# 

# # Model Building

# # Spliting the dataset into Features and Labels

# In[73]:


X = df.drop(['loan_status' , 'status', 'children' , 'family_members', 'mobile'], axis = 1) # Assume that X is our independent variable

y = df['loan_status'] # Assume that y is our dependent variable


# In[74]:


# checking the column names of independent variable(X)
X.columns


# In[75]:




X.head() # previewing the independent variable X


# In[76]:


print(f'The shpape of X ={X.shape} \n') # checking the shape of X

print(f'The shape of y ={y.shape}') # checking the shape of y


# In[77]:


# splitting the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# 
# Observation:
# 
#     We have splitted the dataset into 80% train and 20% test
#     We have used Random Forest Classifier as our model
#     80% of our dataset is used for training
#     20% of our dataset is used for testing
# 

# In[78]:


# instantiating the model
model = RandomForestClassifier()


# In[ ]:





# # Fitting the model

# In[79]:


model.fit(X_train, y_train )


# # Predicting the model

# In[80]:


# making predictions on the test set
y_pred = model.predict(X_test)


# # Actual vs Predicted

# In[81]:




# creating a dataframe to compare the actual and predicted values
pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).head(10)


# # Model Evaluation

# In[82]:


(f'Accuracy Score : {accuracy_score(y_test, y_pred)*100:.2f} %')


# In[83]:


print(f"Precision Score : {precision_score(y_test, y_pred , average = 'micro'):.2f}")


# In[84]:


print(f"F1-Score : {f1_score(y_test, y_pred , average='micro')}")


# In[85]:


# classification report
print(classification_report(y_test, y_pred))


# Interpretation
# 
# Accuracy Score
# 
#     Accuracy score is basically the percentage of correct predictions made by the model out of all the predictions
#         (i.e. the number of correct predictions divided by the total number of predictions)
# 
# The value of accuracy score ranges from 0 to 1 (100%).
# 
# The accuracy score is of our model is 88.11% which is really good
# Precision Score
# 
#     Precision score is basically the percentage of correct positive predictions made by the model
#         Formula: Precision = TP / (TP + FP)
#         ( i.e. the number of correct positive predictions divided by the total number of positive predictions)
#     The value of precision score ranges from 0 to 1.
# 
#     We calculated the precision score of our model using the parameter average = 'micro' which calculates the precision score globally by considering the total number of true positives, false positives, and false negatives across all classes. It treats all instances (samples) equally, regardless of their class labels.
# 
#     The precision score is of our model is 0.88 which indicates that the number of False positives is very less.
# 
# Recall Score
# 
#     Recall is basically the percentage of correct negative predictions made by the model
#         Formula: Recall = TP / (TP + FN)
#         (i.e. the number of correct negative predictions divided by the total number of negative predictions)
# 
#     The value of recall score ranges from 0 to 1.
# 
#     The Recall score is of our model is 0.8811 which indicates that the number of False negatives is very less.
# 
# F1 Score
# 
#     F1 score is basically the harmonic mean of precision and recall
#         Formula: F1 = 2 (precision recall) / (precision + recall)
#     It is usefull when we have imbalanced classes in our dataset and it gives the best results by including precision and recall.
# 

# In[ ]:




