#!/usr/bin/env python
# coding: utf-8

# # **TikTok Project**
# **Course 2 - Get Started with Python**

# Welcome to the TikTok Project!
# 
# You have just started as a data professional at TikTok.
# 
# The team is still in the early stages of the project. You have received notice that TikTok's leadership team has approved the project proposal. To gain clear insights to prepare for a claims classification model, TikTok's provided data must be examined to begin the process of exploratory data analysis (EDA).
# 
# A notebook was structured and prepared to help you in this project. Please complete the following questions.

# # **Course 2 End-of-course project: Inspect and analyze data**
# 
# In this activity, you will examine data provided and prepare it for analysis.
# <br/>
# 
# **The purpose** of this project is to investigate and understand the data provided. This activity will:
# 
# 1.   Acquaint you with the data
# 
# 2.   Compile summary information about the data
# 
# 3.   Begin the process of EDA and reveal insights contained in the data
# 
# 4.   Prepare you for more in-depth EDA, hypothesis testing, and statistical analysis
# 
# **The goal** is to construct a dataframe in Python, perform a cursory inspection of the provided dataset, and inform TikTok data team members of your findings.
# <br/>
# *This activity has three parts:*
# 
# **Part 1:** Understand the situation
# * How can you best prepare to understand and organize the provided TikTok information?
# 
# **Part 2:** Understand the data
# 
# * Create a pandas dataframe for data learning and future exploratory data analysis (EDA) and statistical activities
# 
# * Compile summary information about the data to inform next steps
# 
# **Part 3:** Understand the variables
# 
# * Use insights from your examination of the summary data to guide deeper investigation into variables
# 
# <br/>
# 
# To complete the activity, follow the instructions and answer the questions below. Then, you will us your responses to these questions and the questions included in the Course 2 PACE Strategy Document to create an executive summary.
# 
# Be sure to complete this activity before moving on to Course 3. You can assess your work by comparing the results to a completed exemplar after completing the end-of-course project.

# # **Identify data types and compile summary information**
# 

# Throughout these project notebooks, you'll see references to the problem-solving framework PACE. The following notebook components are labeled with the respective PACE stage: Plan, Analyze, Construct, and Execute.
# 
# # **PACE stages**
# 
# <img src="images/Pace.png" width="100" height="100" align=left>
# 
#    *        [Plan](#scrollTo=psz51YkZVwtN&line=3&uniqifier=1)
#    *        [Analyze](#scrollTo=mA7Mz_SnI8km&line=4&uniqifier=1)
#    *        [Construct](#scrollTo=Lca9c8XON8lc&line=2&uniqifier=1)
#    *        [Execute](#scrollTo=401PgchTPr4E&line=2&uniqifier=1)

# <img src="images/Plan.png" width="100" height="100" align=left>
# 
# 
# ## **PACE: Plan**
# 
# Consider the questions in your PACE Strategy Document and those below to craft your response:
# 
# 

# ### **Task 1. Understand the situation**
# 
# *   How can you best prepare to understand and organize the provided information?
# 
# *Begin by exploring your dataset and consider reviewing the Data Dictionary.*

# I can best prepare for this stage of the project by ensuring that I understand the scope of this stage. 
# -To what extent am I analyzing the data. 
# -What specific data analytics does the management team want? 
# * Min/Max/Mean/Median/std/etc.
# -Certain data may not be relevant to the full scope of the project so I’ll need to determine what is relevant and what isn’t.

# <img src="images/Analyze.png" width="100" height="100" align=left>
# 
# ## **PACE: Analyze**
# 
# Consider the questions in your PACE Strategy Document to reflect on the Analyze stage.

# ### **Task 2a. Imports and data loading**
# 
# Start by importing the packages that you will need to load and explore the dataset. Make sure to use the following import statements:
# *   `import pandas as pd`
# 
# *   `import numpy as np`
# 

# In[2]:


# Import packages
import pandas as pd
import numpy as np


# Then, load the dataset into a dataframe. Creating a dataframe will help you conduct data manipulation, exploratory data analysis (EDA), and statistical activities.
# 
# **Note:** As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[3]:


# Load dataset into dataframe
data = pd.read_csv("tiktok_dataset.csv")


# ### **Task 2b. Understand the data - Inspect the data**
# 
# View and inspect summary information about the dataframe by **coding the following:**
# 
# 1. `data.head(10)`
# 2. `data.info()`
# 3. `data.describe()`
# 
# *Consider the following questions:*
# 
# **Question 1:** When reviewing the first few rows of the dataframe, what do you observe about the data? What does each row represent?
# 
# **Question 2:** When reviewing the `data.info()` output, what do you notice about the different variables? Are there any null values? Are all of the variables numeric? Does anything else stand out?
# 
# **Question 3:** When reviewing the `data.describe()` output, what do you notice about the distributions of each variable? Are there any questionable values? Does it seem that there are outlier values?
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

# In[4]:


# Display and examine the first ten rows of the dataframe
data.head(10)


# In[5]:


# Get summary info
data.info()


# In[6]:


# Get summary statistics
data.describe()


# Question 1: When reviewing the first few rows of the dataframe, what do you observe about the data? What does each row represent?
# 
# Answer 1: The data contains both claim data veriables (claim_status, author_ban_status, verified_status) and video data veriables (video_id, video_duration_sec, video_transcription_text, video_view_count, video_like_count, video_share_count, video_download_count, video_comment_count).
# 
# 
# Question 2: When reviewing the data.info() output, what do you notice about the different variables? Are there any null values? Are all of the variables numeric? Does anything else stand out?
# 
# Answer 2: Judging from the data.info() summary info, there seems to be around 300 more null values on video data veriables listed above than those of the claim data veriables. This will require some investigation.
# 
# Question 3: When reviewing the data.describe() output, what do you notice about the distributions of each variable? Are there any questionable values? Does it seem that there are outlier values?
# 
# Answer 3: There are wildly different videos in this database for claims. For one the min values are intresting due to their extreamly small pool of data. 5 sec vids is extreamly short for analysis. As mentioned in question 2, there appears to be certain veriables that were not tracked due to low numbers but still were counted as a claim. Look at the counts at the top for each item. There are 19382 claims total but only 19084 of them have their views logged. This could scew the data being analyzed. 
# 

# ### **Task 2c. Understand the data - Investigate the variables**
# 
# In this phase, you will begin to investigate the variables more closely to better understand them.
# 
# You know from the project proposal that the ultimate objective is to use machine learning to classify videos as either claims or opinions. A good first step towards understanding the data might therefore be examining the `claim_status` variable. Begin by determining how many videos there are for each different claim status.

# In[7]:


# What are the different values for claim status and how many of each are in the data?
#19,382 / 
data['claim_status'].value_counts()


# **Question:** What do you notice about the values shown?
# There is nearly an even split between the "claim" at 9608 and the "opinion" at 9476.
# There is also only a total of 19084 when you add these values together.

# Next, examine the engagement trends associated with each different claim status.
# 
# Start by using Boolean masking to filter the data according to claim status, then calculate the mean and median view counts for each claim status.

# In[8]:


# What is the average view count of videos with "claim" status?
claim_mask = data['claim_status'] == 'claim'
data[claim_mask]['video_view_count'].agg(['mean', 'median'])


# In[9]:


# What is the average view count of videos with "opinion" status?
opinion_mask = data['claim_status'] == 'opinion'
data[opinion_mask]['video_view_count'].agg(['mean', 'median'])


# **Question:** What do you notice about the mean and media within each claim category?
# 
# Answer: The 'opinions' get significantly less views than those of 'claims'. 
# 
# Now, examine trends associated with the ban status of the author.
# 
# Use `groupby()` to calculate how many videos there are for each combination of categories of claim status and author ban status.

# In[10]:


# Get counts for each group combination of claim status and author ban status
claim_and_author = data.groupby(['claim_status', 'author_ban_status'])
claim_and_author['claim_status'].count()


# **Question:** What do you notice about the number of claims videos with banned authors? Why might this relationship occur?
# 
# Answer: There are significantly more banned and under review claims compared to that of the 'opinons'. This could corrilate to the significant number of views the claims get compared to that of opinions. More eyes on a videos could translate into more cans or under reviews being reported.
# 
# Continue investigating engagement levels, now focusing on `author_ban_status`.
# 
# Calculate the median video share count of each author ban status.

# In[12]:


### YOUR CODE HERE ###
author_median = data.groupby(['author_ban_status'])


# In[13]:


# What's the median video share count of each author ban status?
author_median['video_share_count'].median()


# **Question:** What do you notice about the share count of banned authors, compared to that of active authors? Explore this in more depth.
# 
# Answer: There are significantly more share counts for banned and under review than the active videos. This leads me to further believe my thinking in the last question where more views = the higher likleyhood of a report.
# 
# Use `groupby()` to group the data by `author_ban_status`, then use `agg()` to get the count, mean, and median of each of the following columns:
# * `video_view_count`
# * `video_like_count`
# * `video_share_count`
# 
# Remember, the argument for the `agg()` function is a dictionary whose keys are columns. The values for each column are a list of the calculations you want to perform.

# In[17]:


author_agg = data.groupby(['author_ban_status'])
author_agg['video_view_count','video_like_count','video_share_count'].agg(['count','mean','median'])



# **Question:** What do you notice about the number of views, likes, and shares for banned authors compared to active authors?
# 
# Answer: The views, shares, and the likes for the banned and under review authors are significantly higher compared to those of the active authors. Also note the counts of these author status's. There are significantly higher active authors. Having higher views, shares, and the likes leads me to believe what I suggested previously where you're more likley to get reported.
# 
# Now, create three new columns to help better understand engagement rates:
# * `likes_per_view`: represents the number of likes divided by the number of views for each video
# * `comments_per_view`: represents the number of comments divided by the number of views for each video
# * `shares_per_view`: represents the number of shares divided by the number of views for each video

# In[15]:


# Create a likes_per_view column
data['likes_per_view'] = data['video_like_count'] / data['video_view_count']

# Create a comments_per_view column
data['comments_per_view'] = data['video_comment_count'] / data['video_view_count']

# Create a shares_per_view column
data['shares_per_view'] = data['video_share_count'] / data['video_view_count']
data.head()


# Use `groupby()` to compile the information in each of the three newly created columns for each combination of categories of claim status and author ban status, then use `agg()` to calculate the count, the mean, and the median of each group.

# In[16]:


### YOUR CODE HERE ###
mask_new_columns = data.groupby(['claim_status', 'author_ban_status'])
mask_new_columns['likes_per_view','comments_per_view','shares_per_view'].agg(['count','mean','median'])


# **Question:**
# 
# How does the data for claim videos and opinion videos compare or differ? Consider views, comments, likes, and shares.
# 
# Answer: The 'opinions' get less likes, comments and shares per video. As shown earlier also they also get less views in general. Thus significantly less reports for bans or under reviews. 

# <img src="images/Construct.png" width="100" height="100" align=left>
# 
# ## **PACE: Construct**
# 
# **Note**: The Construct stage does not apply to this workflow. The PACE framework can be adapted to fit the specific requirements of any project.
# 
# 
# 

# <img src="images/Execute.png" width="100" height="100" align=left>
# 
# ## **PACE: Execute**
# 
# Consider the questions in your PACE Strategy Document and those below to craft your response.

# ### **Given your efforts, what can you summarize for Rosie Mae Bradshaw and the TikTok data team?**
# 
# *Note for Learners: Your answer should address TikTok's request for a summary that covers the following points:*
# 
# *   What percentage of the data is comprised of claims and what percentage is comprised of opinions?
# *   What factors correlate with a video's claim status?
# *   What factors correlate with a video's engagement level?
# 

# Question: What percentage of the data is comprised of claims and what percentage is comprised of opinions?
# 
# Answer: 
# Out of 19,382 19084
# claim      9608
# opinion    9476
# Claim account for 49.57% of authors
# Opinion's account for 48.89% of authors
# Unaccounted from total. 
# 
# Question: What factors correlate with a video's claim status?
# 
# Answer: Views, Comments, and shares correlate with a video's claim status. Looking at the data there are significantly more views, comments, and shares on banned and under review authors.
# 
# Question: What factors correlate with a video's engagement level?
# 
# Answer: The claim_status appears to correlate with the videos engagment level. What I mean by this is there are significantly more engagment (likes, shares, and comments) on videos of claims. On opinion videos there aren't as many of any of these factors. This results in more author claims which result in more bans and under reviews. 
# 

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged.
