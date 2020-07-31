#!/usr/bin/env python
# coding: utf-8

# ## Data Classification

# In[1]:


## for system
import sys
import os

## for data
import pandas as pd
import numpy as np

## for plotting
import matplotlib.pyplot as plt
import seaborn as sns

## for statistical tests
import scipy
import statsmodels.formula.api as smf
import statsmodels.api as sm

## for machine learning
from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition
import sklearn as sk
from sklearn.preprocessing import LabelEncoder


# In[2]:


data = pd.read_csv(r'C:\Users\dell\Desktop\CC\DS_DATESET.xlsx')
#data = pd.read_csv(sys.argv[1], header=0)
data.head()


# In[3]:


def utils_recognize_type(data, col, max_cat=20):
    if (data[col].dtype == "O") | (data[col].nunique() < max_cat):
        return "cat"
    else:
        return "num"


# In[4]:


max_cat=20
dic_cols = {col:utils_recognize_type(data, col,max_cat=max_cat) for col in data.columns}
heatmap = data.isnull()
for k,v in dic_cols.items():
    if v == "num":
        heatmap[k] = heatmap[k].apply(lambda x: 0.5 if x is False else 1)
    else:
        heatmap[k] = heatmap[k].apply(lambda x: 0 if x is False else 1)
sns.heatmap(heatmap, cbar=False)
plt.title('Dataset Overview')
#plt.show()
#print("\033[1;37;40m Categerocial ", "\033[1;30;41m Numeric ", "\033[1;30;47m NaN ")


# In[5]:


data = data.rename(columns={'Rate your verbal communication skills [1-10]':'Verbalcommskills','Rate your written communication skills [1-10]':'Writtencommskills','DOB [DD/MM/YYYY]':'DOB','How Did You Hear About This Internship?':'HearAbInternship','Programming Language Known other than Java (one major)':'OneMajorProglang','Which-year are you studying in?':'YearOfstudy','Major/Area of Study':'AreaOfstudy','CGPA/ percentage':'CGPA','Expected Graduation-year':'ExpGradYear','Zip Code':'ZipCode','First Name':'Fname','Last Name':'Lname','Email Address':'EmailId','College name':'ClgName','University Name':'UniName','Course Type':'CourseType','Areas of interest':'AreasOfInterest','Current Employment Status':'CurrentEmployStatus','Have you worked core Java':'WorkedCoreJava','Have you worked on MySQL or Oracle database':'WorkedMySQL','Have you studied OOP Concepts':'StudiedOOPConcepts'})
#data.info()


# In[6]:


data = data.rename(columns={"Label":"Y"})


# In[7]:


y = "Y"
ax = data[y].value_counts().sort_values().plot(kind="barh")
totals= []
for i in ax.patches:
    totals.append(i.get_width())
total = sum(totals)
for i in ax.patches:
     ax.text(i.get_width()+.3, i.get_y()+.20, 
     str(round((i.get_width()/total)*100, 2))+'%', 
     fontsize=10, color='black')
ax.grid(axis="x")
plt.suptitle(y, fontsize=20)
#plt.show()


# In[8]:


cat, num = "Y", "Verbalcommskills"
fig_dim = (10,5)
fig, ax = plt.subplots(nrows=1, ncols=3,  sharex=False, sharey=False, figsize=fig_dim)
fig.suptitle("Rate your verbal communication skills   vs   Y", fontsize=10)
            
### distribution
ax[0].title.set_text('density')
for i in data[cat].unique():
    sns.distplot(data[data[cat]==i][num], hist=False, label=i, ax=ax[0])
ax[0].grid(True)
### stacked
ax[1].title.set_text('bins')
breaks = np.quantile(data[num], q=np.linspace(0,1,20))
tmp = data.groupby([cat, pd.cut(data[num], breaks, duplicates='drop')]).size().unstack().T
tmp = tmp[data[cat].unique()]
tmp["tot"] = tmp.sum(axis=1)
for col in tmp.drop("tot", axis=1).columns:
     tmp[col] = tmp[col] / tmp["tot"]
tmp.drop("tot", axis=1).plot(kind='bar', stacked=True, ax=ax[1], legend=False, grid=True)
### boxplot   
ax[2].title.set_text('outliers')
sns.boxplot(x=cat, y=num, data=data, ax=ax[2])
ax[2].grid(True)
#plt.show()


# In[9]:


cat, num = "Y", "Writtencommskills"
fig_dim = (10,5)
fig, ax = plt.subplots(nrows=1, ncols=3,  sharex=False, sharey=False, figsize=fig_dim)
fig.suptitle("Rate your written communication skills   vs   Y", fontsize=10)
            
### distribution
ax[0].title.set_text('density')
for i in data[cat].unique():
    sns.distplot(data[data[cat]==i][num], hist=False, label=i, ax=ax[0])
ax[0].grid(True)
### stacked
ax[1].title.set_text('bins')
breaks = np.quantile(data[num], q=np.linspace(0,1,20))
tmp = data.groupby([cat, pd.cut(data[num], breaks, duplicates='drop')]).size().unstack().T
tmp = tmp[data[cat].unique()]
tmp["tot"] = tmp.sum(axis=1)
for col in tmp.drop("tot", axis=1).columns:
     tmp[col] = tmp[col] / tmp["tot"]
tmp.drop("tot", axis=1).plot(kind='bar', stacked=True, ax=ax[1], legend=False, grid=True)
### boxplot   
ax[2].title.set_text('outliers')
sns.boxplot(x=cat, y=num, data=data, ax=ax[2])
ax[2].grid(True)
#plt.show()


# In[15]:


cat, num = "Y", "Age"
model = smf.ols(num+' ~ '+cat, data=data).fit()
table = sm.stats.anova_lm(model)
p = table["PR(>F)"][0]
coeff, p = None, round(p, 3)
conclusion = "Correlated" if p < 0.05 else "Non-Correlated"
pvalue = str(p)


# In[16]:


cat, num = "Y", "Writtencommskills"
model = smf.ols(num+' ~ '+cat, data=data).fit()
table = sm.stats.anova_lm(model)
p = table["PR(>F)"][0]
coeff, p = None, round(p, 3)
conclusion = "Correlated" if p < 0.05 else "Non-Correlated"
pvalue = str(p)


# In[17]:


cat, num = "Y", "Verbalcommskills"
model = smf.ols(num+' ~ '+cat, data=data).fit()
table = sm.stats.anova_lm(model)
p = table["PR(>F)"][0]
coeff, p = None, round(p, 3)
conclusion = "Correlated" if p < 0.05 else "Non-Correlated"
pvalue = str(p)


# In[18]:


cat, num = "Y", "CGPA"
model = smf.ols(num+' ~ '+cat, data=data).fit()
table = sm.stats.anova_lm(model)
p = table["PR(>F)"][0]
coeff, p = None, round(p, 3)
conclusion = "Correlated" if p < 0.05 else "Non-Correlated"
pvalue = str(p)


# In[19]:


cat, num = "Y", "ExpGradYear"
model = smf.ols(num+' ~ '+cat, data=data).fit()
table = sm.stats.anova_lm(model)
p = table["PR(>F)"][0]
coeff, p = None, round(p, 3)
conclusion = "Correlated" if p < 0.05 else "Non-Correlated"
pvalue = str(p)


# In[20]:


cat, num = "Y", "ZipCode"
model = smf.ols(num+' ~ '+cat, data=data).fit()
table = sm.stats.anova_lm(model)
p = table["PR(>F)"][0]
coeff, p = None, round(p, 3)
conclusion = "Correlated" if p < 0.05 else "Non-Correlated"
pvalue = str(p)


# In[21]:


x, y = "City", "Y"
cont_table = pd.crosstab(index=data[x], columns=data[y])
chi2_test = scipy.stats.chi2_contingency(cont_table)
chi2, p = chi2_test[0], chi2_test[1]
n = cont_table.sum().sum()
phi2 = chi2/n
r,k = cont_table.shape
phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
rcorr = r-((r-1)**2)/(n-1)
kcorr = k-((k-1)**2)/(n-1)
coeff = np.sqrt(phi2corr/min((kcorr-1), (rcorr-1)))
coeff, p = round(coeff, 3), round(p, 3)
conclusion = "Significant" if p < 0.05 else "Non-Significant"
pvalue = str(p)


# In[22]:


x, y = "AreasOfInterest", "Y"
cont_table = pd.crosstab(index=data[x], columns=data[y])
chi2_test = scipy.stats.chi2_contingency(cont_table)
chi2, p = chi2_test[0], chi2_test[1]
n = cont_table.sum().sum()
phi2 = chi2/n
r,k = cont_table.shape
phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
rcorr = r-((r-1)**2)/(n-1)
kcorr = k-((k-1)**2)/(n-1)
coeff = np.sqrt(phi2corr/min((kcorr-1), (rcorr-1)))
coeff, p = round(coeff, 3), round(p, 3)
conclusion = "Significant" if p < 0.05 else "Non-Significant"
pvalue = str(p)


# In[23]:


x, y = "OneMajorProglang", "Y"
cont_table = pd.crosstab(index=data[x], columns=data[y])
chi2_test = scipy.stats.chi2_contingency(cont_table)
chi2, p = chi2_test[0], chi2_test[1]
n = cont_table.sum().sum()
phi2 = chi2/n
r,k = cont_table.shape
phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
rcorr = r-((r-1)**2)/(n-1)
kcorr = k-((k-1)**2)/(n-1)
coeff = np.sqrt(phi2corr/min((kcorr-1), (rcorr-1)))
coeff, p = round(coeff, 3), round(p, 3)
conclusion = "Significant" if p < 0.05 else "Non-Significant"
pvalue = str(p)


# In[24]:


x, y = "YearOfstudy", "Y"
cont_table = pd.crosstab(index=data[x], columns=data[y])
chi2_test = scipy.stats.chi2_contingency(cont_table)
chi2, p = chi2_test[0], chi2_test[1]
n = cont_table.sum().sum()
phi2 = chi2/n
r,k = cont_table.shape
phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
rcorr = r-((r-1)**2)/(n-1)
kcorr = k-((k-1)**2)/(n-1)
coeff = np.sqrt(phi2corr/min((kcorr-1), (rcorr-1)))
coeff, p = round(coeff, 3), round(p, 3)
conclusion = "Significant" if p < 0.05 else "Non-Significant"
pvalue = str(p)


# In[25]:


x, y = "ClgName", "Y"
cont_table = pd.crosstab(index=data[x], columns=data[y])
chi2_test = scipy.stats.chi2_contingency(cont_table)
chi2, p = chi2_test[0], chi2_test[1]
n = cont_table.sum().sum()
phi2 = chi2/n
r,k = cont_table.shape
phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
rcorr = r-((r-1)**2)/(n-1)
kcorr = k-((k-1)**2)/(n-1)
coeff = np.sqrt(phi2corr/min((kcorr-1), (rcorr-1)))
coeff, p = round(coeff, 3), round(p, 3)
conclusion = "Significant" if p < 0.05 else "Non-Significant"
pvalue = str(p)


# In[26]:


x, y = "Degree", "Y"
cont_table = pd.crosstab(index=data[x], columns=data[y])
chi2_test = scipy.stats.chi2_contingency(cont_table)
chi2, p = chi2_test[0], chi2_test[1]
n = cont_table.sum().sum()
phi2 = chi2/n
r,k = cont_table.shape
phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
rcorr = r-((r-1)**2)/(n-1)
kcorr = k-((k-1)**2)/(n-1)
coeff = np.sqrt(phi2corr/min((kcorr-1), (rcorr-1)))
coeff, p = round(coeff, 3), round(p, 3)
conclusion = "Significant" if p < 0.05 else "Non-Significant"
pvalue = str(p)


# In[22]:


data['Y'] = data['Y'].replace('ineligible',0)
data['Y'] = data['Y'].replace('eligible',1)


# In[23]:


## split data
data_train, data_test = model_selection.train_test_split(data, 
                      test_size=0.3)
## print info
#print("X_train shape:", data_train.drop("Y",axis=1).shape, "| X_test shape:", data_test.drop("Y",axis=1).shape)
#print("y_train mean:", round(np.mean(data_train["Y"]),2), "| y_test mean:", round(np.mean(data_test["Y"]),2))
#print(data_train.shape[1], "features:", data_train.drop("Y",axis=1).columns.to_list())


# In[24]:


data_train = data_train.dropna(axis=1)


# In[25]:


data_train.isna()


# In[26]:


## create dummy
dummy = pd.get_dummies(data_train["AreaOfstudy"], 
                       prefix="AreaOfstudy")
data_train= pd.concat([data_train, dummy], axis=1)
#print( data_train.filter(like="AreaOfstudy", axis=1).head() )
## drop the original categorical column
data_train = data_train.drop("AreaOfstudy", axis=1)
data_train.head()


# In[27]:


## create dummy
dummy = pd.get_dummies(data_train["Gender"], 
                       prefix="Gender")
data_train= pd.concat([data_train, dummy], axis=1)
#print( data_train.filter(like="Gender", axis=1).head())
## drop the original categorical column
data_train = data_train.drop("Gender", axis=1)
data_train.head()


# In[28]:


## create dummy
'''dummy = pd.get_dummies(data_train["Degree"], 
                       prefix="Degree")
data_train= pd.concat([data_train, dummy], axis=1)
print( data_train.filter(like="Degree", axis=1).head() )
## drop the original categorical column
data_train = data_train.drop("Degree", axis=1)
data_train.head()'''


# Label Encode instead of dummy variables

mappings = []

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

data_train = data_train.drop('Degree', axis=1)
for i, col in enumerate(data_train):
    if data_train[col].dtype == 'object':
        data_train[col] = label_encoder.fit_transform(np.array(data_train[col].astype(str)).reshape((-1,)))
        mappings.append(dict(zip(label_encoder.classes_, range(1, len(label_encoder.classes_)+1))))


# In[29]:


## create dummy
dummy = pd.get_dummies(data_train["AreasOfInterest"], 
                       prefix="AreasOfInterest")
data_train= pd.concat([data_train, dummy], axis=1)
#print( data_train.filter(like="AreasOfInterest", axis=1).head())
## drop the original categorical column
data_train = data_train.drop("AreasOfInterest", axis=1)
data_train.head()


# In[30]:


data_train.info()


# In[31]:


## create dummy
dummy = pd.get_dummies(data_train["HearAbInternship"], 
                       prefix="HearAbInternship")
data_train= pd.concat([data_train, dummy], axis=1)
#print( data_train.filter(like="HearAbInternship", axis=1).head())
## drop the original categorical column
data_train = data_train.drop("HearAbInternship", axis=1)
data_train.head()


# In[32]:


#data_train.info()


# In[33]:


## create dummy
dummy = pd.get_dummies(data_train["OneMajorProglang"], 
                       prefix="OneMajorProglang")
data_train= pd.concat([data_train, dummy], axis=1)
#print( data_train.filter(like="OneMajorProglang", axis=1).head())
## drop the original categorical column
data_train = data_train.drop("OneMajorProglang", axis=1)
data_train.head()


# In[34]:


## create dummy
dummy = pd.get_dummies(data_train["City"], 
                       prefix="City")
data_train= pd.concat([data_train, dummy], axis=1)
#print( data_train.filter(like="City", axis=1).head())
## drop the original categorical column
data_train = data_train.drop("City", axis=1)
data_train.head()


# In[35]:


data_train = data_train.reset_index()
data_train.head()


# In[36]:


data_train['YearOfstudy'].value_counts()


# In[37]:


#le=LabelEncoder()
#le.fit_transform(['First-year','Second-year','Third-year','Fourth-year'])
#data_train['YearOfstudy'] = data_train['YearOfstudy'].map({'First-year': 1,'Second-year': 2,'Third-year': 3,'Fourth-year': 4})


# In[38]:


data_train.head()


# In[39]:


data_train = data_train.drop(['Fname','Lname','State','ZipCode','DOB','EmailId','Contact Number','Emergency Contact Number','UniName','CourseType','CurrentEmployStatus','WorkedMySQL','StudiedOOPConcepts','index'], axis=1)
data_train.head()


# In[40]:


#le=LabelEncoder()
#le.fit_transform(['No','Yes'])
#data_train['WorkedCoreJava'] = data_train['WorkedCoreJava'].map({'No': 0,'Yes': 1})
#data_train.head()


# In[41]:


data_train['ClgName'].value_counts()


# In[42]:


data_train = data_train.drop(['ClgName'], axis=1)


# In[43]:


data_train.head()


# In[44]:


scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
X = scaler.fit_transform(data_train.drop("Y", axis=1))
data_scaled= pd.DataFrame(X, columns=data_train.drop("Y", axis=1).columns, index=data_train.index)
data_scaled["Y"] = data_train["Y"]
data_scaled.head()


# In[45]:


corr_matrix = data.copy()
for col in corr_matrix.columns:
    if corr_matrix[col].dtype == "O":
         corr_matrix[col] = corr_matrix[col].factorize(sort=True)[0]
corr_matrix = corr_matrix.corr(method="pearson")
sns.heatmap(corr_matrix, vmin=-1., vmax=1., annot=True, fmt='.2f', cmap="YlGnBu", cbar=True, linewidths=0.5)
plt.title("pearson correlation")


# In[46]:


X = data_train.drop("Y", axis=1).values
y = data_train["Y"].values
feature_names = data_train.drop("Y", axis=1).columns
## Anova
selector = feature_selection.SelectKBest(score_func=  
               feature_selection.f_classif, k=10).fit(X,y)
anova_selected_features = feature_names[selector.get_support()]

## Lasso regularization
selector = feature_selection.SelectFromModel(estimator= 
              linear_model.LogisticRegression(C=1, penalty="l1", 
              solver='liblinear'), max_features=10).fit(X,y)
lasso_selected_features = feature_names[selector.get_support()]
 
## Plot
fig_dims = (10, 10)
fig, ax = plt.subplots(figsize=fig_dims)
data_features = pd.DataFrame({"features":feature_names})
data_features["anova"] = data_features["features"].apply(lambda x: "anova" if x in anova_selected_features else "")
data_features["num1"] = data_features["features"].apply(lambda x: 1 if x in anova_selected_features else 0)
data_features["lasso"] = data_features["features"].apply(lambda x: "lasso" if x in lasso_selected_features else "")
data_features["num2"] = data_features["features"].apply(lambda x: 1 if x in lasso_selected_features else 0)
data_features["method"] = data_features[["anova","lasso"]].apply(lambda x: (x[0]+" "+x[1]).strip(), axis=1)
data_features["selection"] = data_features["num1"] + data_features["num2"]
#sns.barplot(y="features", x="selection", hue="method", data=data_features.sort_values("selection", ascending=False), dodge=False, ax=ax)


# In[47]:


X = data_train.drop("Y", axis=1).values
y = data_train["Y"].values
feature_names = data_train.drop("Y", axis=1).columns.tolist()
## Importance
model = ensemble.RandomForestClassifier(n_estimators=100,
                      criterion="entropy", random_state=0)
model.fit(X,y)
importances = model.feature_importances_
## Put in a pandas dtf
data_importances = pd.DataFrame({"IMPORTANCE":importances, 
            "VARIABLE":feature_names}).sort_values("IMPORTANCE", 
            ascending=False)
data_importances['cumsum'] =  data_importances['IMPORTANCE'].cumsum(axis=0)
data_importances = data_importances.set_index("VARIABLE")
    
## Plot
fig_dim=(30,15)
fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=fig_dim)
fig.suptitle("Features Importance", fontsize=30)
ax[0].title.set_text('variables')
data_importances[["IMPORTANCE"]].sort_values(by="IMPORTANCE").plot(kind="barh", legend=False, ax=ax[0]).grid(axis="x")
ax[0].set(ylabel="")
ax[1].title.set_text('cumulative')
data_importances[["cumsum"]].plot(kind="line", linewidth=4, legend=False, ax=ax[1])
ax[1].set(xlabel="", xticks=np.arange(len(data_importances)), xticklabels=data_importances.index)
plt.xticks(rotation=70)
plt.grid(axis='both')
#plt.show()


# In[48]:


data_test = data_test.dropna(axis=1)


# In[49]:


data_test.isna()


# In[50]:


## create dummy
dummy = pd.get_dummies(data_test["AreaOfstudy"], 
                       prefix="AreaOfstudy")
data_test= pd.concat([data_test, dummy], axis=1)
#print( data_test.filter(like="AreaOfstudy", axis=1).head() )
## drop the original categorical column
data_test = data_test.drop("AreaOfstudy", axis=1)
data_test.head()


# In[51]:


## create dummy
dummy = pd.get_dummies(data_test["Gender"], 
                       prefix="Gender")
data_test= pd.concat([data_test, dummy], axis=1)
#print( data_test.filter(like="Gender", axis=1).head())
## drop the original categorical column
data_test = data_test.drop("Gender", axis=1)
data_test.head()


# In[52]:


## create dummy
'''dummy = pd.get_dummies(data_test["Degree"], 
                       prefix="Degree")
data_test= pd.concat([data_test, dummy], axis=1)
print( data_test.filter(like="Degree", axis=1).head() )
## drop the original categorical column
data_test = data_test.drop("Degree", axis=1)
data_test.head()'''

# Label Encode instead of dummy variables

mappings = []

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

data_test = data_test.drop('Degree', axis=1)
for i, col in enumerate(data_test):
    if data_test[col].dtype == 'object':
        data_test[col] = label_encoder.fit_transform(np.array(data_test[col].astype(str)).reshape((-1,)))
        mappings.append(dict(zip(label_encoder.classes_, range(1, len(label_encoder.classes_)+1))))


# In[53]:


## create dummy
dummy = pd.get_dummies(data_test["AreasOfInterest"], 
                       prefix="AreasOfInterest")
data_test= pd.concat([data_test, dummy], axis=1)
#print( data_test.filter(like="AreasOfInterest", axis=1).head())
## drop the original categorical column
data_test = data_test.drop("AreasOfInterest", axis=1)
data_test.head()


# In[54]:


#data_test.info()


# In[55]:


## create dummy
dummy = pd.get_dummies(data_test["HearAbInternship"], 
                       prefix="HearAbInternship")
data_test = pd.concat([data_test, dummy], axis=1)
#print( data_test.filter(like="HearAbInternship", axis=1).head())
## drop the original categorical column
data_test = data_test.drop("HearAbInternship", axis=1)
data_test.head()


# In[165]:


## create dummy
dummy = pd.get_dummies(data_test["OneMajorProglang"], 
                       prefix="OneMajorProglang")
data_test= pd.concat([data_test, dummy], axis=1)
#print( data_test.filter(like="OneMajorProglang", axis=1).head())
## drop the original categorical column
data_test = data_test.drop("OneMajorProglang", axis=1)
data_test.head()


# In[57]:


## create dummy
dummy = pd.get_dummies(data_test["City"], 
                       prefix="City")
data_test= pd.concat([data_test, dummy], axis=1)
#print( data_test.filter(like="City", axis=1).head())
## drop the original categorical column
data_test = data_test.drop("City", axis=1)
data_test.head()


# In[58]:


data_test = data_test.reset_index()
data_test.head()


# In[59]:


#le=LabelEncoder()
#le.fit_transform(['First-year','Second-year','Third-year','Fourth-year'])
#data_test['YearOfstudy'] = data_test['YearOfstudy'].map({'First-year': 1,'Second-year': 2,'Third-year': 3,'Fourth-year': 4})


# In[60]:


data_test = data_test.drop(['Fname','Lname','State','ZipCode','DOB','EmailId','Contact Number','Emergency Contact Number','UniName','CourseType','CurrentEmployStatus','WorkedMySQL','StudiedOOPConcepts','index','ClgName'], axis=1)
data_test.head()


# In[61]:


#le=LabelEncoder()
#le.fit_transform(['No','Yes'])
#data_test['WorkedCoreJava'] = data_test['WorkedCoreJava'].map({'No': 0,'Yes': 1})
#data_test.head()


# In[62]:


scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
X = scaler.fit_transform(data_test.drop("Y", axis=1))
data_scaled= pd.DataFrame(X, columns=data_test.drop("Y", axis=1).columns, index=data_test.index)
data_scaled["Y"] = data_test["Y"]
data_scaled.head()


# In[63]:


corr_matrix = data.copy()
for col in corr_matrix.columns:
    if corr_matrix[col].dtype == "O":
         corr_matrix[col] = corr_matrix[col].factorize(sort=True)[0]
corr_matrix = corr_matrix.corr(method="pearson")
#sns.heatmap(corr_matrix, vmin=-1., vmax=1., annot=True, fmt='.2f', cmap="YlGnBu", cbar=True, linewidths=0.5)
plt.title("pearson correlation")


# In[64]:


X = data_test.drop("Y", axis=1).values
y = data_test["Y"].values
feature_names = data_test.drop("Y", axis=1).columns
## Anova
selector = feature_selection.SelectKBest(score_func=  
               feature_selection.f_classif, k=10).fit(X,y)
anova_selected_features = feature_names[selector.get_support()]

## Lasso regularization
selector = feature_selection.SelectFromModel(estimator= 
              linear_model.LogisticRegression(C=1, penalty="l1", 
              solver='liblinear'), max_features=10).fit(X,y)
lasso_selected_features = feature_names[selector.get_support()]
 
## Plot
fig_dims = (10, 10)
fig, ax = plt.subplots(figsize=fig_dims)
data_features = pd.DataFrame({"features":feature_names})
data_features["anova"] = data_features["features"].apply(lambda x: "anova" if x in anova_selected_features else "")
data_features["num1"] = data_features["features"].apply(lambda x: 1 if x in anova_selected_features else 0)
data_features["lasso"] = data_features["features"].apply(lambda x: "lasso" if x in lasso_selected_features else "")
data_features["num2"] = data_features["features"].apply(lambda x: 1 if x in lasso_selected_features else 0)
data_features["method"] = data_features[["anova","lasso"]].apply(lambda x: (x[0]+" "+x[1]).strip(), axis=1)
data_features["selection"] = data_features["num1"] + data_features["num2"]
sns.barplot(y="features", x="selection", hue="method", data=data_features.sort_values("selection", ascending=False), dodge=False, ax=ax)


# In[65]:


X = data_test.drop("Y", axis=1).values
y = data_test["Y"].values
feature_names = data_test.drop("Y", axis=1).columns.tolist()
## Importance
model = ensemble.RandomForestClassifier(n_estimators=100,
                      criterion="entropy", random_state=0)
model.fit(X,y)
importances = model.feature_importances_
## Put in a pandas dtf
data_importances = pd.DataFrame({"IMPORTANCE":importances, 
            "VARIABLE":feature_names}).sort_values("IMPORTANCE", 
            ascending=False)
data_importances['cumsum'] =  data_importances['IMPORTANCE'].cumsum(axis=0)
data_importances = data_importances.set_index("VARIABLE")
    
## Plot
fig_dim=(30,15)
fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=fig_dim)
fig.suptitle("Features Importance", fontsize=30)
ax[0].title.set_text('variables')
data_importances[["IMPORTANCE"]].sort_values(by="IMPORTANCE").plot(kind="barh", legend=False, ax=ax[0]).grid(axis="x")
ax[0].set(ylabel="")
ax[1].title.set_text('cumulative')
data_importances[["cumsum"]].plot(kind="line", linewidth=4, legend=False, ax=ax[1])
ax[1].set(xlabel="", xticks=np.arange(len(data_importances)), xticklabels=data_importances.index)
plt.xticks(rotation=70)
plt.grid(axis='both')
#plt.show()


# In[156]:


#X_names = ["Age", "Writtencommskills", "Verbalcommskills", "CGPA", "ExpGradYear", "YearOfstudy",'WorkedCoreJava','AreasOfInterest_IoT','AreaOfstudy_Computer Engineering'] 
#X_names = data_train.reindex(columns=X_names)
#X_train = data_train[X_names].values
#y_train = data_train["Y"].values
#X_test = data_test[X_names].values
#y_test = data_test["Y"].values

X_names = ["Age", "Writtencommskills", "Verbalcommskills", "CGPA", "ExpGradYear", "YearOfstudy",'WorkedCoreJava','AreaOfstudy_Computer Engineering','Gender_Female','City_2','AreasOfInterest_0','HearAbInternship_7']
X_train = data_train[X_names].values
y_train = data_train["Y"].values
X_test = data_test[X_names].values
y_test = data_test["Y"].values


# In[157]:


import numpy as np
cv = model_selection.StratifiedKFold(n_splits=10, shuffle=True)
tprs, aucs = [], []
mean_fpr = np.linspace(0,1,100)
fig = plt.figure()
i = 1
for train, test in cv.split(X_train, y_train):
   prediction = model.fit(X_train[train],y_train[train]).predict_proba(X_train[test])
   fpr, tpr, t = metrics.roc_curve(y_train[test], prediction[:, 1])
   tprs.append(np.interp(mean_fpr, fpr, tpr))
   roc_auc = metrics.auc(fpr, tpr)
   aucs.append(roc_auc)
   plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
   i = i+1
   
plt.plot([0,1], [0,1], linestyle='--', lw=2, color='black')
mean_tpr = np.mean(tprs, axis=0)
mean_auc = metrics.auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='blue', label=r'Mean ROC (AUC = %0.2f )' % (mean_auc), lw=2, alpha=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('K-Fold Validation')
plt.legend(loc="lower right")
plt.show()


# In[158]:


## train
model.fit(X_train, y_train)
## test
predicted_prob = model.predict_proba(X_test)[:,1]
predicted = model.predict(X_test)


# In[159]:


classes = np.unique(y_test)
fig, ax = plt.subplots()
cm = metrics.confusion_matrix(y_test, predicted, labels=classes)
sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False)
ax.set(xlabel="Pred", ylabel="True", title="Confusion matrix")
ax.set_yticklabels(labels=classes, rotation=0)
#plt.show()


# In[160]:


#print("True:", y_test[40], "--> Pred:", predicted[40], "| Prob:", np.max(predicted_prob[40]))


# In[161]:


## for explainer
from lime import lime_tabular

explainer = lime_tabular.LimeTabularExplainer(training_data=X_train, feature_names=X_names, class_names=np.unique(y_train), mode="classification")
explained = explainer.explain_instance(X_test[4], model.predict_proba, num_features=10)
explained.as_pyplot_figure()


# In[162]:


from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


# In[163]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy
print(metrics.accuracy_score(y_test, y_pred))


# In[164]:


# Model Precision: what percentage of positive tuples are labeled as such?
precision = metrics.precision_score(y_test, y_pred)

# Model Recall: what percentage of positive tuples are labelled as such?
recall = metrics.recall_score(y_test, y_pred)

F1 = 2 * (precision * recall) / (precision + recall)
#print(F1)


# ## Logistic Regression

# In[149]:


from sklearn.linear_model import LogisticRegression

# all parameters not specified are set to their defaults
logisticRegr = LogisticRegression()

logisticRegr.fit(X_train, y_train)


# In[150]:


logisticRegr.predict(X_test[0].reshape(1,-1))


# In[151]:


logisticRegr.predict(X_test[0:10])


# In[152]:


predictions = logisticRegr.predict(X_test)


# In[153]:


score = logisticRegr.score(X_test, y_test)
#print(score)


# In[154]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

cm = metrics.confusion_matrix(y_test, predictions)


# In[155]:


plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);

