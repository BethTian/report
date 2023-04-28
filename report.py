#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import numpy as np
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt 
import statsmodels.api as sm


# In[3]:


boston_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv'
boston_df=pd.read_csv(boston_url)


# In[4]:


boston_df


# In[5]:


# Boxplot for the "Median value of owner-occupied homes"
ax =boston_df.boxplot(column = "MEDV")
plt.show()


# In[6]:


#Bar plot for the Charles river variable
ax = sns.barplot(x="CHAS", y="CRIM", data=boston_df)
plt.show()


# In[7]:


boston_df['age_group'] = pd.cut(boston_df['AGE'], bins=[0, 35, 70, float('inf')], labels=['35 or younger', 'between 35 and 70', '70 or older'])

# create a boxplot of MEDV vs age_group
boston_df.boxplot(column='MEDV', by='age_group', figsize=(8,6))
plt.xlabel('Age Group')
plt.ylabel('MEDV')
plt.title('Boxplot of MEDV vs Age Group')
plt.suptitle('')
plt.show()


# In[15]:


ax = sns.scatterplot(x='NOX', y='INDUS', data=boston_df)
plt.show()


# In[14]:


print("They have a positive relationship")


# In[20]:


boston_df.hist(column='PTRATIO')
plt.show()


# In[50]:


#t-test for independent samples
print("null hypothesis : the median values of houses bounded by the Charles river is the same.")

scipy.stats.ttest_ind(boston_df[boston_df['CHAS'] == 1]['MEDV'],
                   boston_df[boston_df['CHAS'] == 0]['MEDV'], equal_var = True)


# **conclusion** :The conclusion is that because the p-value is less than alpha 0.025( two tailed), we reject the null hypothesis that there is a significant difference in median value of houses bounded by the Charles river

# In[57]:


#ANOVA 
print("null hypothethis: Median values of houses (MEDV) for each proportion of owner occupied units built prior to 1940 (AGE) is the same")
groups = boston_df.groupby('AGE')['MEDV'].median()
group1 = groups[0:10].values
group2 = groups[10:20].values
group3 = groups[20:30].values
group4 = groups[30:40].values
group5 = groups[40:50].values
scipy.stats.f_oneway(group1, group2, group3, group4, group5)


# **conclusion** :Because the p-value is more than alpha 0.025, The conclusion is that I cannot reject the null hypothesis that there is a difference  in Median values of houses (MEDV) for each proportion of owner occupied units built prior to 1940 (AGE).

# In[52]:


#Pearson Correlation
print("null hypothesis: the there is no relationship between Nitric oxide concentrations and proportion of non-retail business acres per town")
scipy.stats.pearsonr(boston_df['NOX'], boston_df['INDUS'])


# **conclusion** :I cannot reject the null hypothesis because the p-value is much more that 0.025

# In[54]:


#Regression analysis
## X is the input variables (or independent variables)
X = boston_df['DIS']
## y is the target/dependent variable
y = boston_df['MEDV']
## add an intercept (beta_0) to our model
X = sm.add_constant(X) 

model = sm.OLS(y, X).fit()
predictions = model.predict(X)

# Print out the statistics
model.summary()


# **Conclusion:** Like the t-test, the p-value is less than the alpha (α) level = 0.05, so we reject the null hypothesis as there is evidence that there is a difference in mean evaluation scores based on gender. The coefficient 1.0916 means that MEDV get 1.0916 higher as the DIS increases one unit.
# 

# In[ ]:




