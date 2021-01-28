#!/usr/bin/env python
# coding: utf-8

# ## Analyze A/B Test Results
# 
# You may either submit your notebook through the workspace here, or you may work from your local machine and submit through the next page.  Either way assure that your code passes the project [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).  **Please save regularly.**
# 
# This project will assure you have mastered the subjects covered in the statistics lessons.  The hope is to have this project be as comprehensive of these topics as possible.  Good luck!
# 
# ## Table of Contents
# - [Introduction](#intro)
# - [Part I - Probability](#probability)
# - [Part II - A/B Test](#ab_test)
# - [Part III - Regression](#regression)
# 
# 
# <a id='intro'></a>
# ### Introduction
# 
# A/B tests are very commonly performed by data analysts and data scientists.  It is important that you get some practice working with the difficulties of these 
# 
# For this project, you will be working to understand the results of an A/B test run by an e-commerce website.  Your goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.
# 
# **As you work through this notebook, follow along in the classroom and answer the corresponding quiz questions associated with each question.** The labels for each classroom concept are provided for each question.  This will assure you are on the right track as you work through the project, and you can feel more confident in your final submission meeting the criteria.  As a final check, assure you meet all the criteria on the [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).
# 
# <a id='probability'></a>
# #### Part I - Probability
# 
# To get started, let's import our libraries.

# In[1]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)


# `1.` Now, read in the `ab_data.csv` data. Store it in `df`.  **Use your dataframe to answer the questions in Quiz 1 of the classroom.**
# 
# a. Read in the dataset and take a look at the top few rows here:

# In[2]:


df = pd.read_csv('ab_data.csv')

df.head()


# b. Use the cell below to find the number of rows in the dataset.

# In[3]:


print('The number of rows in the dataset is {} rows'.format(df.shape[0]))


# c. The number of unique users in the dataset.

# In[4]:


print('The number of unique users in the dataset is {}'.format(df.user_id.nunique()))


# d. The proportion of users converted.

# In[5]:


print('The proportion of users converted is {} %'.format(round(df.converted.mean()*100)))


# e. The number of times the `new_page` and `treatment` don't match.

# In[6]:


new_treatment = df[((df['group'] == 'treatment') == (df['landing_page'] == 'new_page')) == False].shape[0]
print('The number of times the new_page and treatment don\'t match are {}'.format(new_treatment))


# f. Do any of the rows have missing values?

# In[7]:


df.isnull().sum()


# `2.` For the rows where **treatment** does not match with **new_page** or **control** does not match with **old_page**, we cannot be sure if this row truly received the new or old page.  Use **Quiz 2** in the classroom to figure out how we should handle these rows.  
# 
# a. Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz.  Store your new dataframe in **df2**.

# In[8]:


df2 = df.copy()

df2 = df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) != False]


# In[9]:


# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# `3.` Use **df2** and the cells below to answer questions for **Quiz3** in the classroom.

# a. How many unique **user_id**s are in **df2**?

# In[10]:


print('The number of unqiue user_id in df2 is {}'.format(df2.user_id.nunique()))


# b. There is one **user_id** repeated in **df2**.  What is it?

# In[11]:


print('The user id that is repeated in df2 is {}'.format(df2[df2.user_id.duplicated()].user_id.iloc[0]))


# c. What is the row information for the repeat **user_id**? 

# In[12]:


df2[df2.user_id == 773192]


# d. Remove **one** of the rows with a duplicate **user_id**, but keep your dataframe as **df2**.

# In[13]:


df2.drop(1899, inplace=True)

df2[df2.user_id == 773192]


# `4.` Use **df2** in the cells below to answer the quiz questions related to **Quiz 4** in the classroom.
# 
# a. What is the probability of an individual converting regardless of the page they receive?

# In[14]:


print('The probability of an individual converting regardless of the page they receive is {}'.format(df2['converted'].mean()))


# b. Given that an individual was in the `control` group, what is the probability they converted?

# In[15]:


control_df = df2.query('group == "control"')

print('The probability of an individual in control group converting is {}'.format(control_df['converted'].mean()))


# c. Given that an individual was in the `treatment` group, what is the probability they converted?

# In[16]:


treatment_df = df2.query('group == "treatment"')

print('The probability of an individual in treatment group converting is {}'.format(treatment_df['converted'].mean()))


# d. What is the probability that an individual received the new page?

# In[17]:


(df2['landing_page'] == "new_page").mean()

print('The probability that an individual received the new page is {}.'.format((df2['landing_page'] == "new_page").mean()))


# e. Consider your results from parts (a) through (d) above, and explain below whether you think there is sufficient evidence to conclude that the new treatment page leads to more conversions.

# Answer : There is no sufficient evidence to conclude that the new treatment page leads to more conversions since the probability individual converting in either control or treatment group is almost similiar, what makes it even against the new treatment page is that the probability of the old-control page conversion is more than the new-treatment one by .001578.
# 

# <a id='ab_test'></a>
# ### Part II - A/B Test
# 
# Notice that because of the time stamp associated with each event, you could technically run a hypothesis test continuously as each observation was observed.  
# 
# However, then the hard question is do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time?  How long do you run to render a decision that neither page is better than another?  
# 
# These questions are the difficult parts associated with A/B tests in general.  
# 
# 
# `1.` For now, consider you need to make the decision just based on all the data provided.  If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should your null and alternative hypotheses be?  You can state your hypothesis in terms of words or in terms of **$p_{old}$** and **$p_{new}$**, which are the converted rates for the old and new pages.

# Answer:The null and alternative hypotheses are as follows 
# 
# H0 = **$p_{new}$** - **$p_{old}$** <= 0
# 
# H1 = **$p_{new}$** - **$p_{old}$** > 0
# 

# `2.` Assume under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, assume they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br><br>
# 
# Use a sample size for each page equal to the ones in **ab_data.csv**.  <br><br>
# 
# Perform the sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null.  <br><br>
# 
# Use the cells below to provide the necessary parts of this simulation.  If this doesn't make complete sense right now, don't worry - you are going to work through the problems below to complete this problem.  You can use **Quiz 5** in the classroom to make sure you are on the right track.<br><br>

# a. What is the **conversion rate** for $p_{new}$ under the null? 

# In[18]:


p_new = df2['converted'].mean()

print('The conversion rate for P𝑛𝑒𝑤 is {}'.format(p_new))


# b. What is the **conversion rate** for $p_{old}$ under the null? <br><br>

# In[19]:


p_old = df2['converted'].mean()

print('The conversion rate for P𝑜𝑙𝑑 is {}'.format(p_old))


# c. What is $n_{new}$, the number of individuals in the treatment group?

# In[20]:


n_new = treatment_df['converted'].count()

print('The number of individuals in the treatment group is {}'.format(n_new))


# d. What is $n_{old}$, the number of individuals in the control group?

# In[21]:


n_old = control_df['converted'].count()

print('The number of individuals in the treatment group is {}'.format(n_old))


# e. Simulate $n_{new}$ transactions with a conversion rate of $p_{new}$ under the null.  Store these $n_{new}$ 1's and 0's in **new_page_converted**.

# In[22]:


new_page_converted = np.random.binomial(n_new, p_new)
new_page_converted


# f. Simulate $n_{old}$ transactions with a conversion rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.

# In[23]:


old_page_converted = np.random.binomial(n_old, p_old)
old_page_converted


# g. Find $p_{new}$ - $p_{old}$ for your simulated values from part (e) and (f).

# In[24]:


diff = (new_page_converted/n_new) - (old_page_converted/n_old)
diff


# h. Create 10,000 $p_{new}$ - $p_{old}$ values using the same simulation process you used in parts (a) through (g) above. Store all 10,000 values in a NumPy array called **p_diffs**.

# In[25]:


p_diffs = []
for _ in range(10000):
    new_converted = np.random.binomial(n_new, p_new)/n_new
    old_converted = np.random.binomial(n_old, p_old)/n_old
    p_diffs.append(new_converted - old_converted)


# i. Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.

# In[27]:


plt.hist(p_diffs);


# j. What proportion of the **p_diffs** are greater than the actual difference observed in **ab_data.csv**?

# In[28]:


#actual obs_diff in ab_data
act_old_mean = df.query('group == "control"').converted.mean()
act_new_mean = df.query('group == "treatment"').converted.mean()
act_diff = act_new_mean - act_old_mean

p_diffs = np.array(p_diffs)

#propotion of p_diffs greater than actual act_diff
(p_diffs > act_diff).mean()


# In[29]:


plt.hist(p_diffs);
plt.axvline(act_diff, c = 'red');


# k. Please explain using the vocabulary you've learned in this course what you just computed in part **j.**  What is this value called in scientific studies?  What does this value mean in terms of whether or not there is a difference between the new and old pages?

# **Answer :** The value calculated here is the P-value of the observation of this statistic giving the null condition is true, the value means that we would fail to reject the null condition and then keep the old page because the P-value is large.

# l. We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let `n_old` and `n_new` refer the the number of rows associated with the old page and new pages, respectively.

# In[30]:


import statsmodels.api as sm
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

convert_old = control_df.query('converted == 1')['converted'].count()
convert_new = treatment_df.query('converted == 1')['converted'].count()
n_old = control_df['converted'].count()
n_new = treatment_df['converted'].count()
convert_old, convert_new, n_old, n_new


# m. Now use `stats.proportions_ztest` to compute your test statistic and p-value.  [Here](https://docs.w3cub.com/statsmodels/generated/statsmodels.stats.proportion.proportions_ztest/) is a helpful link on using the built in.

# In[31]:


z_score, p_vals = sm.stats.proportions_ztest([convert_old, convert_new], [n_old, n_new], alternative ='smaller')
z_score, p_vals


# n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?

# **Answer:** The z-score means that the difference between the test statistic and the null hypothesis is 1.31 standard deviations above the mean, where our p_value is 0.905 which is not below $\alpha$ = 0.05 and will lead us to reject the null hypothesis.

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# `1.` In this final part, you will see that the result you achieved in the A/B test in Part II above can also be achieved by performing regression.<br><br> 
# 
# a. Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

# **Answer:** It will be a logistic regression because we are dealing with binary output.

# b. The goal is to use **statsmodels** to fit the regression model you specified in part **a.** to see if there is a significant difference in conversion based on which page a customer receives. However, you first need to create in df2 a column for the intercept, and create a dummy variable column for which page each user received.  Add an **intercept** column, as well as an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.

# In[32]:


df2['intercept'] = 1
df2[['ab_page', 'old_page']] = pd.get_dummies(df2['landing_page'])
df2.head()


# c. Use **statsmodels** to instantiate your regression model on the two columns you created in part b., then fit the model using the two columns you created in part **b.** to predict whether or not an individual converts. 

# In[33]:


logit_mod = sm.Logit(df2['converted'], df2[['intercept', 'ab_page']])
results = logit_mod.fit()


# d. Provide the summary of your model below, and use it as necessary to answer the following questions.

# In[34]:


results.summary()


# e. What is the p-value associated with **ab_page**? Why does it differ from the value you found in **Part II**?<br><br>  **Hint**: What are the null and alternative hypotheses associated with your regression model, and how do they compare to the null and alternative hypotheses in **Part II**?

# 
# **Answer** : The p-value calculated here is 0.190. This is because the Logistic Regression is based on a two-tailed test. 0.190 is still greater then 0.05 (our $\alpha$), so we still cannot reject our null hypothesis, the null hypothesis is that when ab_page = 1, converted = 0; the alternative hypothesis is that when ab_page = 1, converted is more likely to be 1.
# 

# f. Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

# **Answer**: The convert rate might be influenced by some users features that aren't mentioned in the dataset, adding them might reveal hidden values of the new version of the page.

# g. Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives in. You will need to read in the **countries.csv** dataset and merge together your datasets on the appropriate rows.  [Here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html) are the docs for joining tables. 
# 
# Does it appear that country had an impact on conversion?  Don't forget to create dummy variables for these country columns - **Hint: You will need two columns for the three dummy variables.** Provide the statistical output as well as a written response to answer this question.

# In[35]:


countries = pd.read_csv('./countries.csv')
df_new = countries.set_index('user_id').join(df2.set_index('user_id'), how='inner')
df_new.head()


# In[36]:


df_new['country'].value_counts()


# In[37]:


#creating the necessary dummy variable 
df_new[['CA', 'UK', 'US']] = pd.get_dummies(df_new['country'])
df_new.head()


# In[38]:


#Training the model
log_mod = sm.Logit(df_new['converted'], df_new[['intercept', 'UK', 'US']])
result = log_mod.fit()
result.summary()


# **Answer:** Based of the P-values, it doesn't seem that adding new feature as country affects the conversion significantly

# h. Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion.  Create the necessary additional columns, and fit the new model.  
# 
# Provide the summary results, and your conclusions based on the results.

# In[39]:


# Create additional columns specifying what user/country converted
df_new['US_page'] = df_new['US'] * df_new['ab_page']
df_new['UK_page'] = df_new['UK'] * df_new['ab_page']

df_new.head()


# In[40]:


#Training the model
log_mod = sm.Logit(df_new['converted'], df_new[['intercept', 'ab_page', 'US', 'UK', 'US_page', 'UK_page']])
result = log_mod.fit()
result.summary()


# **Answer:** Seems like the P-values are higher than 0.05 which leads us that the interactions between the page and country has no significant effect on conversion.

# <a id='conclusions'></a>
# ## Conclusions
# 
# > The conclusion of the study including all its parts from hypothesis test to logistic regression would lead us to stick with the old page, because we don't have sufficient evidence to reject it as all test were favoured for the old page no matter adding new features like country of user that might affect or enhance the test results, this doesn't mean that adding more features wouldn't affect the rest results, it might do but it will require more time and resource to be spent ot get a final decision.
