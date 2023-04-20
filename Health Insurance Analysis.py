#!/usr/bin/env python
# coding: utf-8

# # Healthcare Insurance Analysis
# 
# ## Problem statement:
# A significant public health concern is the rising cost of healthcare. Therefore, it's crucial to be able to predict future costs and gain a solid understanding of their causes. The insurance industry must also take this analysis seriously. This analysis may be used by
# healthcare insurance providers to make a variety of strategic and tactical decisions.
# 
# ## Objective:
# The objective of this project is to predict patients’ healthcare costs and to identify factors contributing to this prediction. It will also be useful to learn the interdependencies of different factors and comprehend the significance of various tools at various stages of
# the healthcare cost prediction process.
# 
# ## Project Task: Week 1
# Data science/data analysis	
# * 1.	Collate the files so that all the information is in one place
# * 2.	Check for missing values in the dataset
# * 3.	Find the percentage of rows that have trivial value (for example, ?), and delete such rows if they do not contain  significant information
# * 4.	Use the necessary transformation methods to deal with the nominal and ordinal categorical variables in the dataset
# * 5.	The dataset has State ID, which has around 16 states. All states are not represented in equal proportions in the data. Creating dummy variables for all regions may also result in too many insignificant predictors. Nevertheless, only R1011, R1012, and R1013 are worth investigating further. Create a suitable strategy to create dummy variables with these restraints.
# * 6.	The variable NumberOfMajorSurgeries also appears to have string values. Apply a suitable method to clean up this variable.
#   Note: Use Excel as well as Python to complete the tasks
# * 7.	Age appears to be a significant factor in this analysis. Calculate the patients' ages based on their dates of birth.
# * 8.	The gender of the patient may be an important factor in determining the cost of hospitalization. The salutations in a * beneficiary's name can be used to determine their gender. Make a new field for the beneficiary's gender.
# * 9.	You should also visualize the distribution of costs using a histogram, box and whisker plot, and swarm plot.
# * 10.	State how the distribution is different across gender and tiers of hospitals
# * 11.	Create a radar chart to showcase the median hospitalization cost for each tier of hospitals
# * 12.	Create a frequency table and a stacked bar chart to visualize the count of people in the different tiers of cities and hospitals
# * 13.	Test the following null hypotheses:
#    * a.	The average hospitalization costs for the three types of hospitals are not significantly different
#    * b.	The average hospitalization costs for the three types of cities are not significantly different
#    * c.	The average hospitalization cost for smokers is not significantly different from the average cost for nonsmokers
#    * d.	Smoking and heart issues are independent
#      Note: Use Excel as well as Python to complete the tasks
#  
# ## Project Task: Week 2
# Machine Learning	
# * 1.	Examine the correlation between predictors to identify highly correlated predictors. Use a heatmap to visualize this.
# * 2.	Develop and evaluate the final model using regression with a stochastic gradient descent optimizer. Also, ensure that you apply all the following suggestions:
#     Note:
#    •	Perform the stratified 5-fold cross-validation technique for model building and validation
#    •	Use standardization and hyperparameter tuning effectively
#    •	Use sklearn-pipelines
#    •	Use appropriate regularization techniques to address the bias-variance trade-off
#    * a.	Create five folds in the data, and introduce a variable to identify the folds
#    * b.	For each fold, run a for loop and ensure that 80 percent of the data is used to train the model and the remaining 20 percent is used to validate it in each iteration
#    * c.	Develop five distinct models and five distinct validation scores (root mean squared error values)
#    * d.	Determine the variable importance scores, and identify the redundant variables
# * 3.	Use random forest and extreme gradient boosting for cost prediction, share your cross- validation results, and calculate the variable importance scores
# * 4.	Case scenario: Estimate the cost of hospitalization for Christopher, Ms. Jayna (her date of birth is 12/28/1988, height is 170 cm, and weight is 85 kgs). She lives in a tier-1 city and her state’s State ID is R1011. She lives with her partner and two children. She was found to be nondiabetic (HbA1c = 5.8). She smokes but is otherwise healthy. She has had no transplants or major surgeries. Her father died of lung cancer. Hospitalization costs will be estimated using tier-1 hospitals.
# * 5.	Find the predicted hospitalization cost using all five models. The predicted value should be the mean of the five models' predicted values.
# 

# # Week 1
# ## 1. Collate the files so that all the information is in one place

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')


# In[2]:


hospitalisation = pd.read_csv('Hospitalisation details.csv')
medical = pd.read_csv('Medical Examinations.csv')
names = pd.read_excel('Names.xlsx')


# In[3]:


hospitalisation.head()


# In[4]:


hospitalisation.shape


# In[5]:


hospitalisation['Customer ID'].value_counts()


# In[6]:


hospitalisation.sort_values("Customer ID")


# In[7]:


hospitalisation = hospitalisation.sort_values("Customer ID")


# In[8]:


hospitalisation.head()


# In[9]:


hospitalisation['Customer ID'] = hospitalisation['Customer ID'].str.upper()
hospitalisation.head(10)


# In[10]:


medical.head()


# In[11]:


medical['Customer ID'] = medical['Customer ID'].str.upper()
medical.head(10)


# In[12]:


names.head()


# In[13]:


names['Customer ID'] = names['Customer ID'].str.upper()
names.head(10)


# In[14]:


df = pd.merge(hospitalisation, medical, how='outer', on = 'Customer ID')
df


# In[15]:


hospitalisation.shape


# In[16]:


medical.shape


# In[17]:


df.shape


# ## 2. Check for missing values in the dataset.

# In[18]:


hospitalisation.isna().sum()


# In[19]:


df.isna().sum()


# In[20]:


df = pd.merge(df, names, how='outer', on = 'Customer ID')
df


# In[21]:


df.isna().sum()


# In[22]:


df.head()


# In[23]:


df.tail()


# ## 3. Find the percentage of rows that have trivial value (for example, ?), and delete such rows if they do not contain significant information

# In[24]:


df.loc[df['Customer ID']=='?']


# In[25]:


len(df.loc[df['Customer ID']=='?'])


# In[26]:


len(df)


# In[27]:


len(df.loc[df['Customer ID']=='?'])/len(df)*100


# In[28]:


columns = df.columns
for col in columns:
    print(col)
    print(len(df.loc[df[col]=='?'])/len(df)*100)
    print("===================================")


# In[ ]:





# ## 4. Use the necessary transformation methods to deal with the nominal and ordinal categorical variables in the dataset

# In[29]:


df['Customer ID'].value_counts()


# In[30]:


le = LabelEncoder()


# In[31]:


df['Customer ID'] = le.fit_transform(df['Customer ID'])


# In[32]:


df.head()


# In[33]:


df['year'].value_counts()


# In[34]:


df['year'] = df['year'].replace('?',np.nan).astype('float')


# In[35]:


df.isna().sum()


# In[36]:


df['year'].median()


# In[37]:


df['year'] = df['year'].fillna(df['year'].median())


# In[38]:


df.isna().sum()


# In[39]:


df['month'].value_counts()


# In[40]:


df['month'] = df['month'].replace('?',np.nan)


# In[41]:


df['month'].value_counts()


# In[42]:


df['month'].mode()


# In[43]:


df['month'].mode()[0]


# In[44]:


df['month'] = df['month'].fillna(df['month'].mode()[0])


# In[45]:


df['month'].value_counts()


# In[46]:


df['month'] = le.fit_transform(df['month'])


# In[47]:


df['month'].value_counts()


# In[48]:


df['date'].value_counts()


# In[49]:


df['children'].value_counts()


# In[50]:


df.isna().sum()


# In[51]:


df['charges'].value_counts()


# In[52]:


df['Hospital tier'].value_counts()


# In[53]:


df['Hospital tier'].mode()


# In[54]:


df['Hospital tier'].mode()[0]


# In[55]:


df['Hospital tier'] = df['Hospital tier'].replace('?',df['Hospital tier'].mode()[0])


# In[56]:


df['Hospital tier'].value_counts()


# In[57]:


df['City tier'].value_counts()


# In[58]:


df['City tier'].mode()[0]


# In[59]:


df['City tier'] = df['City tier'].replace('?',df['City tier'].mode()[0])


# In[60]:


df['City tier'].value_counts()


# In[61]:


df['State ID'].value_counts()


# In[62]:


df['State ID'].mode()[0]


# In[63]:


df['State ID'] = df['State ID'].replace('?',df['State ID'].mode()[0])


# In[64]:


df['State ID'].value_counts()


# In[65]:


df.isna().sum()


# In[66]:


df['BMI'].value_counts()


# In[67]:


df['BMI'].mean()


# In[68]:


df['BMI'] = df['BMI'].fillna(df['BMI'].mean())


# In[69]:


df.isna().sum()


# In[70]:


df['HBA1C'].value_counts()


# In[71]:


df['HBA1C'] = df['HBA1C'].fillna(df['HBA1C'].mean())


# In[72]:


df.isna().sum()


# In[73]:


df['Heart Issues'].value_counts()


# In[74]:


df['Heart Issues'].mode()[0]


# In[75]:


df['Heart Issues'] = df['Heart Issues'].fillna(df['Heart Issues'].mode()[0])


# In[76]:


df.isna().sum()


# In[77]:


df['Any Transplants'].value_counts()


# In[78]:


df['Any Transplants'] = df['Any Transplants'].fillna(df['Any Transplants'].mode()[0])


# In[79]:


df.isna().sum()


# In[80]:


df['Cancer history'].value_counts()


# In[81]:


df['Cancer history'] = df['Cancer history'].fillna(df['Cancer history'].mode()[0])


# In[ ]:





# ## 6. The variable NumberOfMajorSurgeries also appears to have string values. Apply a suitable method to clean up this variable.

# In[82]:


df['NumberOfMajorSurgeries'].value_counts()


# In[83]:


df['NumberOfMajorSurgeries'] = df['NumberOfMajorSurgeries'].replace('No major surgery',0)


# In[84]:


df['NumberOfMajorSurgeries'].value_counts()


# In[85]:


df['NumberOfMajorSurgeries'].mode()[0]


# In[86]:


df['NumberOfMajorSurgeries'] = df['NumberOfMajorSurgeries'].fillna(df['NumberOfMajorSurgeries'].mode()[0])


# In[87]:


df.isna().sum()


# In[88]:


df['smoker'].value_counts()


# In[89]:


df['smoker'].mode()[0]


# In[90]:


df['smoker'] = df['smoker'].replace(df['smoker'].mode()[0],0)


# In[91]:


df['smoker'] = df['smoker'].fillna(df['smoker'].mode()[0])


# In[92]:


df.isna().sum()


# In[93]:


df['name'].value_counts()


# In[94]:


df['name'] = df['name'].fillna('?')


# In[95]:


df.isna().sum()


# In[96]:


df.head(10)


# In[97]:


df.info()


# In[98]:


df['Hospital tier'].value_counts()


# In[99]:


Hospital = pd.get_dummies(df['Hospital tier'],prefix="Hospital",drop_first=True)


# In[100]:


Hospital


# In[101]:


df['Hospital tier'].value_counts()


# In[102]:


df['City tier'].value_counts()


# In[103]:


City = pd.get_dummies(df['City tier'],prefix="City",drop_first=True)


# In[104]:


City


# In[105]:


df['State ID'].value_counts()


# In[106]:


df['State ID'] = le.fit_transform(df['State ID'])


# In[107]:


df['State ID'].value_counts()


# In[108]:


df['Heart Issues'].value_counts()


# In[109]:


df['Heart Issues'] = le.fit_transform(df['Heart Issues'])


# In[110]:


df['Heart Issues'].value_counts()


# In[111]:


df.info()


# In[112]:


df['Any Transplants'].value_counts()


# In[113]:


df['Any Transplants'] = le.fit_transform(df['Any Transplants'])


# In[114]:


df['Any Transplants'].value_counts()


# In[115]:


df['Cancer history'].value_counts()


# In[116]:


df['Cancer history'] = le.fit_transform(df['Cancer history'])


# In[117]:


df['Cancer history'].value_counts()


# In[118]:


df['NumberOfMajorSurgeries'].value_counts()


# In[119]:


df['NumberOfMajorSurgeries'] = pd.to_numeric(df['NumberOfMajorSurgeries'])


# In[120]:


df.info()


# In[121]:


df['smoker'].value_counts()


# In[122]:


df['smoker'] = df['smoker'].replace('?',0)


# In[123]:


df['smoker'].value_counts()


# In[124]:


df['smoker'] = df['smoker'].replace('yes',1)


# In[125]:


df['smoker'].value_counts()


# ## 5. The dataset has State ID, which has around 16 states. All states are not represented in equal proportions in the data. Creating dummy variables for all regions may also result in too many insignificant predictors. Nevertheless, only R1011, R1012, and R1013 are worth investigating further. Create a suitable strategy to create dummy variables with these restraints.

# In[126]:


df.info()


# In[127]:


df['State ID'].value_counts()


# In[128]:


column = list(['State ID'])
column


# In[129]:


df['State ID'] = df['State ID'].apply(lambda x: 3 if x > 2 else x)


# In[130]:


df['State ID'].value_counts()


# In[131]:


df['State ID'] = df['State ID'].replace([2, 1, 0, 3], ['R1013','R1012','R1011','Other'])


# In[132]:


df['State ID'].value_counts()


# In[133]:


df.head()


# In[ ]:





# ## 7. Age appears to be a significant factor in this analysis. Calculate the patients' ages based on their dates of birth.

# In[134]:


df['age'] = 2023-df['year']


# In[135]:


df.head(10)


# In[136]:


State_ID = pd.get_dummies(df['State ID'],prefix="State_ID", prefix_sep = '_',drop_first=True)
State_ID


# In[137]:


df = pd.concat([df,State_ID],axis=1)
df.head()


# In[138]:


df.head(10)


# In[139]:


df['name'] = df['name'].replace('?',np.nan)


# In[140]:


df.isna().sum()


# In[141]:


df = df.dropna()


# In[142]:


df.shape


# In[143]:


df.head()


# In[144]:


df = df.drop('State ID',axis=1)


# In[145]:


df.head()


# ## 8. The gender of the patient may be an important factor in determining the cost of hospitalization. The salutations in a beneficiary's name can be used to determine their gender. Make a new field for the beneficiary's gender.

# In[146]:


data = df.copy()
data.head()


# In[147]:


data = df[['name']]
data


# In[148]:


new = data["name"].str.split(",", n = 1, expand = True)


# In[149]:


data["First Name"]= new[1]
data["Last Name"]= new[0]
data.head()


# In[150]:


new = data["First Name"].str.split(".", n = 1, expand = True)
data["First Name"]= new[1]
data["gender"]= new[0]
data.head()


# In[151]:


data['gender'].value_counts()


# In[152]:


data['gender']= data['gender'].replace(['Mr','Ms','Mrs'],['male','female','female'])


# In[153]:


data['gender'].value_counts()


# In[154]:


data['name'] = data['First Name'] + " " + data['Last Name']
data['name']


# In[155]:


data.head()


# In[156]:


data = data[['gender','name']]
data.head()


# In[157]:


data.shape


# In[158]:


df.shape


# In[159]:


df.head()


# In[160]:


df = df.drop('name',axis=1)
df.head()


# In[161]:


df = pd.concat([df,data],axis=1)
df.head()


# In[162]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[163]:


df.head()


# In[164]:


df['gender'].unique()


# In[165]:


df['gender'] = df['gender'].str.strip()


# In[166]:


df['gender'].unique()


# In[167]:


df['gender'].value_counts()


# In[168]:


df['gender'] = df['gender'].replace(['Mr','Ms','Mrs'],['male','female','female'])


# In[169]:


df['gender'].value_counts()


# In[170]:


df['gender'] = le.fit_transform(df['gender'])


# In[171]:


df['gender'].value_counts()


# In[172]:


df.head()


# In[173]:


df = pd.concat([df,City,Hospital],axis=1)
df.head()


# In[174]:


final = df.drop(['Hospital tier','City tier','name'],axis=1)
final.head()


# In[175]:


final.shape


# ## 9. You should also visualize the distribution of costs using a histogram, box and whisker plot, and swarm plot.

# In[176]:


df.head()


# In[177]:


plt.figure(figsize=(10,4))
sns.distplot(df['charges'], hist=True, kde=True, bins=500)


# In[178]:


sns.barplot(data=df, x="charges", y="BMI")


# In[179]:


sns.barplot(data=df, x="charges", y="HBA1C")


# In[180]:


plt.figure(figsize=(15,8))
plt.xticks(rotation = 45)
sns.boxplot('City tier', 'charges', data=df)


# In[181]:


plt.figure(figsize=(12,6))
sns.swarmplot('month', 'charges', data=df)


# ## 10. State how the distribution is different across gender and tiers of hospitals.

# In[182]:


sns.countplot('Hospital tier', data=df, hue='gender')


# In[183]:


plt.figure(figsize=(10,4))
sns.displot(data= df, x= "charges", col= "gender")


# In[290]:


sns.histplot(binwidth=1,
            x="smoker",
            hue="Heart Issues",
            data=df,
            multiple="dodge")


# In[293]:


gkk = df.groupby(['smoker', 'Heart Issues']).charges.mean()
gkk


# In[ ]:





# In[184]:


plt.figure(figsize=(10,5))
sns.displot(data= df, x= "charges", col= "Hospital tier")


# In[185]:


plt.figure(figsize=(10,5))
sns.displot(data= df, x= "charges", col= "City tier")


# ## 11. Create a radar chart to showcase the median hospitalization cost for each tier of hospitals

# In[186]:


df[['charges']].groupby(df['Hospital tier']).median()


# In[187]:


hospital_tier = ['tier_1','tier_2','tier_3']
charges = [32097.435,7188.605,10665.440]


# In[188]:


angles=np.linspace(0,2*np.pi,len(hospital_tier), endpoint=False)
print(angles)


# In[189]:


angles=np.concatenate((angles,[angles[0]]))
print(angles)


# In[190]:


hospital_tier.append(hospital_tier[0])
charges.append(charges[0])


# In[191]:


fig=plt.figure(figsize=(6,6))
ax=fig.add_subplot(polar=True)
#basic plot
ax.plot(angles,charges, 'o--', color='g', label='charges')
#fill plot
ax.fill(angles, charges, alpha=0.25, color='g')
#Add labels
ax.set_thetagrids(angles * 180/np.pi, hospital_tier)
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()


# In[ ]:





# ## 12. Create a frequency table and a stacked bar chart to visualize the count of people in the different tiers of cities and hospitals.

# In[192]:


freq_table = pd.crosstab(df['City tier'], df['Hospital tier'])


# In[193]:


freq_table


# In[194]:


df = df.dropna()
df.shape


# In[195]:


freq_table2 = pd.crosstab(df['City tier'], df['gender'])
freq_table2


# In[196]:


freq_table3 = pd.crosstab(df['Hospital tier'], df['gender'])
freq_table3


# In[197]:


freq_table.plot(kind="bar", figsize=(10,6), stacked=True, colormap='Set1')


# In[198]:


freq_table = pd.crosstab(df['Hospital tier'], df['City tier'])
freq_table


# In[199]:


freq_table.plot(kind="bar", figsize=(10,6), stacked=True, colormap='Set1')


# In[200]:


df.head()


# In[201]:


df.columns


# ## 13. Test the following null hypotheses:
# * a.	The average hospitalization costs for the three types of hospitals are not significantly different
# * b.	The average hospitalization costs for the three types of cities are not significantly different
# * c.	The average hospitalization cost for smokers is not significantly different from the average cost for nonsmokers
# * d.	Smoking and heart issues are independent

# In[202]:


from scipy.stats import ttest_1samp,ttest_ind,mannwhitneyu,levene,shapiro
from statsmodels.stats.power import ttest_power


# ### 13.(a) The average hospitalization costs for the three types of hospitals are not significantly different 

# In[203]:


Ho = "The average hospitalization costs for the three types of hospitals are not significantly different"
Ha = "The average hospitalization costs for the three types of hospitals are different"


# In[204]:


x = np.array(df[df['Hospital_tier - 2']==1].charges)
x.shape


# In[205]:


x


# In[206]:


y=np.array(df[df['Hospital_tier - 2']==0].charges)
y


# In[207]:


len(y)


# In[208]:


t,p_value = ttest_ind(x,y,axis=0)


# In[209]:


p_value


# In[210]:


if p_value<0.05:
    print(f'{Ha} as the p_value({p_value})<0.05')
else:
    print(f'{Ho} as the p_value({p_value})>0.05')


# ### 13.(b) The average hospitalization costs for the three types of cities are not significantly different

# In[211]:


Ho = "The average hospitalization costs for the three types of cities are not significantly different"
Ha = "The average hospitalization costs for the three types of cities different"


# In[212]:


x = np.array(df[df['City_tier - 2']==1].charges)
x.shape


# In[213]:


x


# In[214]:


y=np.array(df[df['City_tier - 2']==0].charges)
y


# In[215]:


len(y)


# In[216]:


t,p_value = ttest_ind(x,y,axis=0)


# In[217]:


p_value


# In[218]:


if p_value<0.05:
    print(f'{Ha} as the p_value({p_value})<0.05')
else:
    print(f'{Ho} as the p_value({p_value})>0.05')


# ### 13.(c) The average hospitalization cost for smokers is not significantly different from the average cost for nonsmokers

# In[219]:


Ho = "The average hospitalization cost for smokers is not significantly different from the average cost for nonsmokers"
Ha = "The average hospitalization cost for smokers is different from the average cost for nonsmokers"


# In[220]:


x = np.array(df[df['smoker']==1].charges)
x.shape


# In[221]:


x


# In[222]:


y=np.array(df[df['City_tier - 2']==0].charges)
y


# In[223]:


len(y)


# In[224]:


t,p_value = ttest_ind(x,y,axis=0)


# In[225]:


p_value


# In[226]:


if p_value<0.05:
    print(f'{Ha} as the p_value({p_value})<0.05')
else:
    print(f'{Ho} as the p_value({p_value})>0.05')


# ### 13.(d) Smoking and heart issues are independent

# In[227]:


Ho = "Smoking and heart issues are independent"
Ha = "Smoking and heart issues are dependent"


# In[228]:


x = np.array(df[df['Heart Issues']==1].smoker)
x.shape


# In[229]:


x


# In[230]:


y=np.array(df[df['Heart Issues']==0].smoker)
y


# In[231]:


len(y)


# In[232]:


t,p_value = ttest_ind(x,y,axis=0)


# In[233]:


p_value


# In[234]:


if p_value<0.05:
    print(f'{Ha} as the p_value({p_value})<0.05')
else:
    print(f'{Ho} as the p_value({p_value})>0.05')


# In[ ]:





# In[ ]:





# # Week 2 - Machine Learning

# ## 1. Examine the correlation between predictors to identify highly correlated predictors. Use a heatmap to visualize this.

# In[235]:


df.head()


# In[236]:


df.isna().sum()


# In[237]:


corr_df = df.drop(['City_tier - 2','City_tier - 3','Hospital_tier - 2','Hospital_tier - 3','name'],axis=1)


# In[238]:


corr = corr_df.corr()
corr


# In[239]:


plt.figure(figsize=(15,15))
plt.title('Correlation_Matrix',fontsize=20)
sns.heatmap(corr,cmap='RdYlGn',fmt='g',annot=True,vmax=1.0,vmin=-1.0)
plt.show()


# In[240]:


plt.figure(figsize=(15,7))
plt.title('Correlation Matrix',fontsize=20)
sns.heatmap(corr[['charges']],cmap='RdYlGn',vmax=1.0,vmin=-1.0,fmt='g',annot=True)
plt.show()


# ## 2. Develop and evaluate the final model using regression with a stochastic gradient descent optimizer. Also, ensure that you apply all the following suggestions: 
# * Note: 
# * Perform the stratified 5-fold cross-validation technique for model building and validation 
# * Use standardization and hyperparameter tuning effectively 
# * Use sklearn-pipelines 
# * Use appropriate regularization techniques to address the bias-variance trade-off
# * (a) Create five folds in the data, and introduce a variable to identify the folds
# * (b) For each fold, run a for loop and ensure that 80 percent of the data is used to train the model and the remaining 20            percent is used to validate it in each iteration
# * (c) Develop five distinct models and five distinct validation scores (root mean squared error values)
# * (d) Determine the variable importance scores, and identify the redundant variables

# In[241]:


df.head()


# In[242]:


df.tail()


# In[243]:


final.head()


# In[244]:


final.shape


# In[245]:


final = final.drop('Customer ID',axis=1)
final.head()


# In[246]:


from sklearn.preprocessing import MinMaxScaler


# In[247]:


minmax = MinMaxScaler()


# In[248]:


final_scaled = minmax.fit_transform(final)
final_scaled = pd.DataFrame(final_scaled,columns=final.columns)
final_scaled


# In[249]:


variance = final_scaled.var()
columns = final_scaled.columns


# In[250]:


variable = [ ]

for i in range(0,len(variance)):
    if variance[i]>=0.006: #setting the threshold as 1%
        variable.append(columns[i])
variable


# In[251]:


len(variable)


# In[252]:


final.head()


# In[253]:


final = final.dropna()
final.isna().sum()


# In[254]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score


# In[255]:


X = final.drop('charges',axis=1)
y = final['charges']


# In[256]:


X


# In[257]:


y


# In[258]:


X = np.array(X)
y = np.array(y)


# In[259]:


scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(X)


# In[260]:


x_scaled


# In[ ]:





# In[261]:


from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# In[262]:


pipeline_sgdr=Pipeline([('sgd_regressor',RandomForestRegressor(random_state=0))])


# In[263]:


pipeline_randomforest=Pipeline([('rf_regressor',RandomForestRegressor())])


# In[264]:


pipeline_xgboost=Pipeline([('xgb_regressor',XGBRegressor())])


# In[265]:


## LEts make the list of pipelines
pipelines = [pipeline_sgdr, pipeline_randomforest, pipeline_xgboost]
best_accuracy=0.0
best_regressor=0
best_pipeline=""


# In[266]:


pipe_dict = {0: 'SGD Regressor', 1: 'RandomForest Regressor', 2: 'XGBoost Regressor'}


# In[267]:


from sklearn.model_selection import KFold
skf = KFold(n_splits=5, shuffle=True, random_state=10)
lst_accu_stratified = []

for train_index, test_index in skf.split(x_scaled, y):
    x_train_fold, x_test_fold = x_scaled[train_index], x_scaled[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]
    for pipe in pipelines:
        pipe.fit(x_train_fold, y_train_fold)
        lst_accu_stratified.append(pipe.score(x_test_fold, y_test_fold)) 


# In[268]:


lst_accu_stratified


# In[269]:


for i,model in enumerate(pipelines):
    print("{} Test Accuracy: {}".format(pipe_dict[i],model.score(x_test_fold,y_test_fold)))


# In[270]:


for i,model in enumerate(pipelines):
    if model.score(x_test_fold,y_test_fold)>best_accuracy:
        best_accuracy=model.score(x_test_fold,y_test_fold)
        best_pipeline=model
        best_regressor=i
print('Regressor with best accuracy:{}'.format(pipe_dict[best_regressor]))


# In[271]:


scores = cross_val_score(pipeline_sgdr, x_train_fold, y_train_fold, cv=5)


# In[272]:


scores


# In[273]:


scores.mean()


# In[274]:


scores = cross_val_score(pipeline_randomforest, x_train_fold, y_train_fold, cv=5)
scores


# In[275]:


scores.mean()


# In[276]:


scores = cross_val_score(pipeline_xgboost, x_train_fold, y_train_fold, cv=5)
scores


# In[277]:


scores.mean()


# In[278]:


sgdr = SGDRegressor()
sgdr.fit(x_train_fold, y_train_fold)
sgdr.score(x_test_fold, y_test_fold)


# In[279]:


rfr = RandomForestRegressor()
# fit the model
rfr.fit(x_train_fold, y_train_fold)
print(rfr.score(x_test_fold, y_test_fold))
# get importance
importance = rfr.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()


# In[280]:


xgbr = XGBRegressor()
# fit the model
xgbr.fit(x_train_fold, y_train_fold)
print(xgbr.score(x_test_fold, y_test_fold))
# get importance
importance = xgbr.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
 print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()


# ## 4. Case scenario:
# Estimate the cost of hospitalization for Christopher, Ms. Jayna (her date of birth is 12/28/1988, height is 170 cm, and weight is 85 kgs). She lives in a tier-1 city and her state’s State ID is R1011. She lives with her partner and two children. She was found to be nondiabetic (HbA1c = 5.8). She smokes but is otherwise healthy. She has had no transplants or major surgeries. Her father died of lung cancer. Hospitalization costs will be estimated using tier-1 hospitals.

# In[281]:


pred1 = rfr.predict([[1988,1,28,2,29.4,5.8,0,0,1,0,1,35,0,1,0,0,0,0,0,0]])
pred1


# In[282]:


pred2 = xgbr.predict([[1988,1,28,2,29.4,5.8,0,0,1,0,1,35,0,1,0,0,0,0,0,0]])
pred2


# In[283]:


pred = (pred1+pred2)/2
pred


# ## 5. Find the predicted hospitalization cost using all five models. The predicted value should be the mean of the five models' predicted values.
# 

# In[284]:


sgdr_pred = sgdr.predict(x_test_fold)
sgdr_pred


# In[285]:


rfr_pred = rfr.predict(x_test_fold)
rfr_pred


# In[286]:


xgbr_pred = xgbr.predict(x_test_fold)
xgbr_pred


# In[287]:


final_pred = (sgdr_pred+rfr_pred+xgbr_pred)/3
final_pred

