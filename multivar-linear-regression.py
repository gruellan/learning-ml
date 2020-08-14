import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from word2number import w2n
import math 

# Let's predict salaries based on 3 features: experience, test score and interview score
# Our equation will be: price (y) = (m1 * exp) + (m2 * test_score) + (m3 * int_score) + c

# --- DATA PRE-PROCESSING ---
df = pd.read_csv("hiring.csv")

# Convert text numbers to int numbers
df.experience = df.experience.fillna("zero")
df.experience = df.experience.apply(w2n.word_to_num)

# Set NaNs to median values
median_test = math.floor(df['test_score(out of 10)'].mean())
df.test_score = df.fillna(median_test, inplace=True)
median_interview = math.floor(df['interview_score(out of 10)'].mean())
df.interview_score = df.fillna(median_interview, inplace=True)


# Create model and feed in our data
lr = LinearRegression()
lr.fit(df[['experience', 'test_score(out of 10)', 'interview_score(out of 10)']], df['salary($)'])

print("Predicted salary for employee with 2 years experience, 9/10 test and 6/10 interview:", lr.predict([[2, 9, 6]]))

print("Predicted salary for employee with 12 years experience, 10/10 test and 10/10 interview:", lr.predict([[12, 10, 10]]))