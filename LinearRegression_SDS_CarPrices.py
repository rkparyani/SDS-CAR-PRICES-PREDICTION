# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 16:58:02 2018

@author: errit
"""

# Import Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%matplotlib inline
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
import warnings
from scipy import stats
from scipy.stats import norm, skew #for some statistics

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points
# for ML Modelling

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC, LinearRegression
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error

# Define Classess and Methods



# Declare Variables
featurelist = ['symboling',
'normalized-losses',
'make',
'fuel-type',
'aspiration',
'num-of-doors',
'body-style',
'drive-wheels',
'engine-location',
'wheel-base',
'length',
'width',
'height',
'curb-weight',
'engine-type',
'num-of-cylinders',
'engine-size',
'fuel-system',
'bore',
'stroke',
'compression-ratio',
'horsepower',
'peak-rpm',
'city-mpg',
'highway-mpg',
'price']

# Load the data
carprice = pd.read_csv("automobile.csv",names=featurelist)
carprice.head()

carprice.head().iloc[:,0:14]
carprice.head().iloc[:,14:]


print((carprice=='?').sum())
carprice = carprice.replace("?",np.NaN)
carprice.isna().sum()
cp = carprice.dropna()

cp.dtypes

cp['normalized-losses'] = cp['normalized-losses'].astype(float)
cp['horsepower'] = cp['horsepower'].astype(float)
cp['peak-rpm'] = cp['peak-rpm'].astype(float)
cp['bore'] = cp['bore'].astype(float)
cp['stroke'] = cp['stroke'].astype(float)
cp['price'] = cp['price'].astype(float)


cp.columns = cp.columns.str.replace("-", "")
cp.columns


# EDA

# Analysis on Predictor variable
plt.figure(figsize=(20,20))
sns.distplot(cp['price'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(cp['price'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('price dist')

#Get also the QQ-plot

fig = plt.figure(figsize=(20,20))
res = stats.probplot(cp['price'], plot=plt)
plt.show()

# EDA

# Analysis on Predictor variable

#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
cp["price_log"] = np.log1p(cp["price"])

#Check the new distribution 
sns.distplot(cp['price_log'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(cp['price_log'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('Price')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(cp['price_log'], plot=plt)
plt.show()


numerical_feats = cp.dtypes[cp.dtypes != "object"].index
print("Number of Numerical features: ", len(numerical_feats))

categorical_feats = cp.dtypes[cp.dtypes == "object"].index
print("Number of Categorical features: ", len(categorical_feats))


for col in numerical_feats:
    print('****************')
    print(col)
    print("Skewness: %f" % cp[col].skew())
    print("Kurtosis: %f" % cp[col].kurt())
    
    
    
# Tranform variables with high skewness and Kurtosis

sns.distplot(cp['compressionratio']);
#skewness and kurtosis
print("Skewness: %f" % cp['compressionratio'].skew())
print("Kurtosis: %f" % cp['compressionratio'].kurt())

# Tranform variables with high skewness and Kurtosis

sns.distplot(cp['enginesize']);
#skewness and kurtosis
print("Skewness: %f" % cp['enginesize'].skew())
print("Kurtosis: %f" % cp['enginesize'].kurt())


cp['enginesize_log'] = np.log(cp['enginesize'])
sns.distplot(cp['enginesize_log']);


numerical_feats = cp.dtypes[cp.dtypes != "object"].index
print("Number of Numerical features: ", len(numerical_feats))

categorical_feats = cp.dtypes[cp.dtypes == "object"].index
print("Number of Categorical features: ", len(categorical_feats))



nr_rows = 5
nr_cols = 4

fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*3.5,nr_rows*3))

li_num_feats = list(numerical_feats)
li_not_plot = ['price', 'price_log']
li_plot_num_feats = [c for c in list(numerical_feats) if c not in li_not_plot]


for r in range(0,nr_rows):
    for c in range(0,nr_cols):  
        i = r*nr_cols+c
        if i < len(li_plot_num_feats):
            sns.regplot(cp[li_plot_num_feats[i]], cp['price_log'], ax = axs[r][c])
            stp = stats.pearsonr(cp[li_plot_num_feats[i]], cp['price_log'])
            #axs[r][c].text(0.4,0.9,"title",fontsize=7)
            str_title = "r = " + "{0:.2f}".format(stp[0]) + "      " "p = " + "{0:.2f}".format(stp[1])
            axs[r][c].set_title(str_title,fontsize=11)
            
plt.tight_layout()    
plt.show()   



#Correlation map to see how features are correlated with SalePrice
corrmat = cp.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)


cor = cp.corr()
cor

cor_val = cor['price_log'].abs().sort_values(ascending = False)
cor_df = pd.DataFrame({'Cor' :cor_val})
cor_df


f, ax = plt.subplots(figsize=(5, 4))
plt.xticks(rotation='90')
sns.barplot(x=cor_val.index, y=cor_val)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Correlation with Price', fontsize=15)
plt.title('Continuous Predictor Strength', fontsize=15)


for catg in list(categorical_feats) :
    print(cp[catg].value_counts())
    print('#'*50)


li_cat_feats = list(categorical_feats)
nr_rows = 2
nr_cols = 5

fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*4,nr_rows*3))

for r in range(0,nr_rows):
    for c in range(0,nr_cols):  
        i = r*nr_cols+c
        if i < len(li_cat_feats):
            sns.boxplot(x=li_cat_feats[i], y='price_log', data=cp, ax = axs[r][c])
    
plt.tight_layout()    
plt.show()   


catg_strong_corr = [ 'make', 'aspiration', 'bodystyle', 'drivewheels', 'enginetype', 
                     'numofcylinders','fuelsystem']

catg_weak_corr = ['fultype', 'numofdoors', 'enginelocation']


num_strong_corr = [ 'curbweight', 'enginesize_log', 'width', 'length', 'horsepower', 
                     'higwaympg','citympg','wheelbase','bore']

num_weak_corr = ['height', 'normallizedlosses', 'symboling', 'compressionratio','weakrpm','stroke']



for catg in catg_strong_corr :
    print('********************************')
    g = cp.groupby(catg)['price_log'].mean()
    print(g)




# 'make'
make_catg2 = ['audi','bmw','volvo','peugot','saab']
make_catg3 = ['porsche','mercedes-benz','jaguar'] 


# aspiration
aspirn_catg2 = ['turbo']


# bodystyle
bdystl_catg2 = ['sedan','wagon','hardtop']
bdystl_catg3 = ['convertible'] 

# drivewheels
drwls_catg2 = ['rwd']
                
# enginetype
engtyp_catg2 = ['dohc','ohcv','l']
                
# numofcylinders
numcyl_catg2 = ['five','six']
numcyl_catg3 = ['eight']
                

# fuelsystem
fuelsys_catg2 = ['mpfi','mfi','idi']    



for df in [cp]:
    
    cp['make_num'] = 1  
    cp.loc[(cp['make'].isin(make_catg2) ), 'make_num'] = 2    
    cp.loc[(cp['make'].isin(make_catg3) ), 'make_num'] = 3        

    cp['aspiration_num'] = 1  
    cp.loc[(cp['aspiration'].isin(aspirn_catg2) ), 'aspiration_num'] = 2    

    cp['bodystyle_num'] = 1  
    cp.loc[(cp['bodystyle'].isin(bdystl_catg2) ), 'bodystyle_num'] = 2    
    cp.loc[(cp['bodystyle'].isin(bdystl_catg3) ), 'bodystyle_num'] = 3    

    cp['drivewheels_num'] = 1  
    cp.loc[(cp['drivewheels'].isin(drwls_catg2) ), 'drivewheels_num'] = 2    

    cp['enginetype_num'] = 1  
    cp.loc[(cp['enginetype'].isin(engtyp_catg2) ), 'enginetype_num'] = 2    
    
    cp['numofcylinders_num'] = 1  
    cp.loc[(cp['numofcylinders'].isin(numcyl_catg2) ), 'numofcylinders_num'] = 2    
    cp.loc[(cp['numofcylinders'].isin(numcyl_catg3) ), 'numofcylinders_num'] = 3    
                
    cp['fuelsystem_num'] = 1  
    cp.loc[(cp['fuelsystem'].isin(fuelsys_catg2) ), 'fuelsystem_num'] = 2    



cp_df =cp.loc[:,['curbweight', 'enginesize_log', 'width', 'length', 'horsepower','highwaympg','citympg','wheelbase','bore','make_num','aspiration_num','bodystyle_num',
          'drivewheels_num','enginetype_num','numofcylinders_num','fuelsystem_num','price_log']]


cp_df.iloc[:,:]



nr_rows = 5
nr_cols = 4
num_feats = cp_df.columns

fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*3.5,nr_rows*3))

li_num_feats = list(num_feats)
li_not_plot = ['price_log']
li_plot_num_feats = [c for c in list(num_feats) if c not in li_not_plot]


for r in range(0,nr_rows):
    for c in range(0,nr_cols):  
        i = r*nr_cols+c
        if i < len(li_plot_num_feats):
            sns.regplot(cp_df[li_plot_num_feats[i]], cp_df['price_log'], ax = axs[r][c])
            stp = stats.pearsonr(cp_df[li_plot_num_feats[i]], cp_df['price_log'])
            #axs[r][c].text(0.4,0.9,"title",fontsize=7)
            str_title = "r = " + "{0:.2f}".format(stp[0]) + "      " "p = " + "{0:.2f}".format(stp[1])
            axs[r][c].set_title(str_title,fontsize=11)
            
plt.tight_layout()    
plt.show()   


plt.subplots(1, 1, figsize=(15,15))
sns.boxplot('bodystyle', y='price_log', data=cp)
    
plt.tight_layout()    
plt.show()   


cp = cp.drop(cp.index[(cp['make']=='dodge') & (cp['price_log']>9.25)])
cp = cp.drop(cp.index[(cp['make']=='dodge') & (cp['price_log']>9.25)])
cp = cp.drop(cp.index[(cp['make']=='toyota') & (cp['price_log']>9.71)])
cp = cp.drop(cp.index[(cp['bodystyle']=='hardtop') & (cp['price_log']>10.1)])

numeric_feats = cp_df.dtypes[cp.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = cp_df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness

skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))


from scipy.special import boxcox1p
skewed_features = skewness.index

lam = 0.15
for feat in skewed_features:
    cp_df[feat] = boxcox1p(cp_df[feat], lam)
    
# Standard Scaler

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
cp_sc = sc.fit_transform(cp_df)



cp_df_sc = pd.DataFrame(cp_sc)

cp_df_sc.head()

cp_df_sc.shape

cp_df_sc.describe()

x = cp_df_sc
x.head()

x.columns
y = cp_df_sc.iloc[:,16:17]
y.head()

train,test = train_test_split(cp_df_sc,shuffle = True, test_size = 0.2,random_state = 88)

train.head()
y_train = train.iloc[:,16:17]
y_train.head()

x_train = train.iloc[:,0:16]
x_train.head()

y_test = test.iloc[:,16:17]
x_test = test.iloc[:,0:16]


lrm = LinearRegression()
model = lrm.fit(x_train,y_train)
y_pred_train = model.predict(x_train)

from sklearn.metrics import r2_score
y_pred_test = model.predict(x_test)
r2_score(y_test,y_pred_test)


model.coef_

r2_score(y_train,y_pred_train)
model.coef_
actual_data = np.array(y_test)

for i in range(len(y_pred_train)-1):
    actual = actual_data[i]
    predicted = y_pred_test[i]
    explained = ((actual_data[i] - y_pred_test[i])/actual_data[i])*100
    print(actual,predicted,explained)

y_pred_test    


    

    