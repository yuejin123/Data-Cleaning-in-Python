# Read data

1. What are the columns and types? Change dtype if necessary

```python
data = pd.read_excel("data.xlsx")
data = pd.ExcelFile('reshaping_data.xlsx')
data.sheet_names
data.parse(sheetname='ABC_inc', skiprows=0).head(10)

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(‘movie_metadata.csv’, 
	dtype={‘duration’: int, "title_year": str},
	names=names)

data = np.loadtxt("data.csv", delimiter=',', skiprows=2)

df.info()
```

2. Indexing?

```python
df.index.values
df.set_index('column_name_to_use', inplace=True)
# shuffle the data
result.reindex(np.random.permutation(result.index))
```
3. Correlation

```python
use_data.corr()['isFraud'].sort_values()
```


# Clean data
Check percentage of problematic data

## Drop unwanted columns
```python
df.drop(columns_to_drop, inplace=True, axis=1)
```
## Outliers

```python
# remove all rows with outliers in at least one row
df = df[(np.abs(stats.zscore(df.drop(['DATE'], axis=1))) < 3).all(axis=1)]
# clip
a.iloc[:,0].apply(lambda x: min(x,5))

```
## Uniqueness

```python
df.iloc[:,0].is_unique

list_columns = []
for i in data.select_dtypes('object'):
    num_unique = len(data[i].unique())
    print(i,' ',len(data[i].unique()))
    if num_unique <20 and num_unique>1: list_columns.append(i)

for i in list_columns:
    print(i,' ',data[i].unique())

```

Dispose duplicates:

```python
# Identify duplicates
from itertools import groupby
res = [(k,sum(1 for i in g)) for k,g in groupby([1,1,1,2,2,3])]
sorted(res,key=lambda x: x[1],reverse=True)[0][1]
## columns that could be different for duplicated transactions
ignored_columns = ['transactionDateTime','currentBalance','availableMoney']
# select all the data that have duplicated and sort
filtered = data.loc[data.drop(ignored_columns,axis=1).duplicated(keep=False),].sort_values(['accountNumber','customerId','merchantName','transactionAmount','transactionDateTime'])

## select duplicated transactions with the previous transaction happening within 23 hours ago
# Mark duplicates as ``True`` except for the first occurrence.
filtered['is_duplicated'] = data.drop(ignored_columns,axis=1).duplicated(keep='first')
filtered['prev_time'] = np.insert(filtered.transactionDateTime.iloc[:-1,].values,0,pd.to_datetime('2099-01-01'))

use_data['accountOpenDate'],\
use_data['currentExpDate'],\
use_data['dateOfLastAddressChange'] = \
    pd.to_datetime(use_data['accountOpenDate']),\
    pd.to_datetime(use_data['currentExpDate']),\
    pd.to_datetime(use_data['dateOfLastAddressChange'])

duplicate_filter =(filtered.transactionDateTime-filtered.prev_time<pd.Timedelta('23h')) & filtered.is_duplicated
```

## Missing values
**Detection**
Pandas will recognize both empty cells and “NA” types as missing values, but might miss some other user-created symbols for missing values('0' or 'na')

```python
missing_values = ["n/a", "na", "--"]
df = pd.read_csv("property data.csv", na_values = missing_values)
df.isnull().sum()/ len(df)
```

When the type of the value is not consistent with the rest of the values

```python
# Detecting numbers 
def is_int(x):
	try: 
        int(s)
        return np.nan
    except ValueError:
        return x

df['OWN_OCCUPIED'].apply(is_int)

for row in df['OWN_OCCUPIED']:
    try:
        int(row)
        df.loc[cnt, 'OWN_OCCUPIED']=np.nan
    except ValueError:
        pass
    cnt+=1
```
**Profile missing values**

```python
df.isnull().values.any()
```
**Handle missing values**

* Binary features: -1 for negative, 0 for missing, 1 for positive
* Numeric features:
- interpolate
- Tree-based methods: encode as a big positive or negative number, 2*max(x), 2*min(x), -99999
- Linear, neural nets: encode by splitting into 2 columns: 1. Binary column indicating NA(0 if not and 1 if yes) 2. Original column replace NAs with mean or median
* Categorical features:
- Encode as an unique category
- Replace with the most frequent category level

```python
df['ST_NUM'].fillna(125, inplace=True)
# limit to fill only one value
df.fillna(method='pad/ffill/bfill', limit=1)
# drop columns which have at least 90% non-NaNs
df.dropna(thresh=int(df.shape[0] * .9), axis=1)
df.dropna(axis=1)
data.dropna(subset=[‘title_year’])

data2 = data1.interpolate(method='linear')
```


# Process data
## Encode



Categorical Encoding: turn categorical features into numeric features to provide more fine-grained information; most ML tools only accept numbers as their input

- Label encoding: good for tree-based methods
- one-hot encoding: good for k-means, linear NNs
- Frequency encoding: 0-1 based on their relative frequencies	
- Target mean encoding: use the mean of the target value for the given level as the encoded value; weighted average of the overall mean of the training set and the mean of the level
- Target mean encoding with smoothing/leave-one-out schema: leave one value in a given level out when encoding
- Target mean with expanding mean schema

Encoding numerical features:

- Binning with quantiles: replace with bin's mean or median; treat each bin as a category level & use any categorical encoding schema
- Dimensionality reduction: SVD & PCA
- Clustering

Target Transformation

- log(x), log(x+1), sqrt(x), sqrt(x+1) etc

```python
# np.where(if_this_condition_is_true, do_this, else_this)
df['new_column'] = np.where(df[i] > 10, 'foo', 'bar')
df['new_column'] = np.where(df['col'].str.startswith('foo') and  
                            not df['col'].str.endswith('bar'), 
                            True, 
                            df['col'])

df['new_column'] = df['numeric'].cut(4)
```

Using sklearn encoders

```python
le = preprocessing.LabelEncoder()
use_data[use_data.select_dtypes(['object','bool']).columns] = use_data.select_dtypes(['object','bool']).apply(le.fit_transform,axis=0)
# one-hot encoder in pandas
pd.get_dummies(df)

```
## Normalize
Sklearn scaler

```python
from sklearn.preprocessing import MinMaxScaler, Binarizer
mm_scaler = MinMaxScaler(feature_range=(0, 1)) # scale to 0 and 1
tz = Binarizer(threshold=-25.0).fit([ary_int])

def log_normalize(series):
    return series.apply(lambda x:math.log(x+1.0))

def clip(series, clip_to_min, clip_to_max):
    return series.apply(lambda x:(min(max(x, clip_to_min), clip_to_max)))

def z_score_normalize(series):
    mean = series.mean()
    std_dv = series.std()
    return series.apply(lambda x:(x - mean) / std_dv)

```

## Reindex & merge
```python
# assign new columns 
df.assign(a=1)
# move 'DATE' column to the front
cols = list(umcsi)
cols.insert(0, cols.pop(cols.index('DATE')))
umcsi = umcsi.reindex(columns = cols)
umcsi.head()
# concatenate all dataframes into one final dataframe  
dfs = [dow,unemp,oil,hstarts,cars,retail,fedrate,umcsi]
# we perform the joins on DATE column as key and drop null values
# reset_index and set_index before merging if join on indices
df = reduce(lambda left,right: pd.merge(left,right,on='DATE', how='outer'), dfs).dropna() 
df.merge(df2)
```

## Grouping
**Melting**

```python
# all the identifier columns to be included in the multi-index 
idx =['district','province','partner','financing_source'
,'main_organization']
pd.melt(df, id_vars=idx, value_vars=['B', 'C'])
```

**Pivoting**

```Python
# Pivoting
example = pd.DataFrame({
    'team':["team %d" % (x+1) for x in range(5)]*5,
    'batting avg': np.random.uniform(.200, .400, 25)
})

example.pivot_table(values='batting avg',columns='team',aggfunc=np.average)
```

**Groupby**

```python
df.groupby('col2').agg({'col1': 'mean'})
```

**Resample and rolling**
resample is more general than asfreq. For example, using resample I can pass an arbitrary function to perform binning over a Series or DataFrame object in bins of arbitrary size(essentially **groupby**). asfreq is a concise way of changing the frequency of a DatetimeIndex object. It also provides padding functionality.

```python
import matplotlib.pyplot as plt
tesla.Open.plot(alpha=0.5, style='-')
# business annual, BQ - business quarter
tesla.Open.resample('3BA').mean().plot(style=':')
tesla.Open.asfreq('BA',method='pad').plot(style='--')

ROI = 100 * (tesla.Close.tshift(-365) / tesla.Close - 1)

rolling = tesla.Open.rolling(365, center=True)

data = pd.DataFrame({'input': tesla['Open'],
                     'one-year rolling_mean': rolling.mean(),
                     'one-year rolling_std': rolling.std()})
ax = data.plot(style=['-', '--', ':'])
ax.lines[0].set_alpha(0.3)

```

## Format
**String formatting**

regex = r'^(\d{4})'(find any four digits at the beginning of a string, which suffices for our case.)


```python
# Using regex
df['Date of Publication'].str.extract(r'^(\d{4})', expand=False)
# strip and sub
def preProcess(column):
    '''
    Used to prevent errors during the dedupe process.
    '''
    try : 
        column = column.decode('utf8')
    except AttributeError:
        pass
    column = unidecode(column)
    column = re.sub('  +', ' ', column)
    column = re.sub('\n', ' ', column)
    column = column.strip().strip('"').strip("'").lower().strip()
    
    if not column:
        column = None
    return column


df.apply(preProcess,axis=0)

```

Use `fuzzywuzzy` to test the similarity of strings

**Datetime formatting**

```python
index = date + pd.to_timedelta(np.arange(12), 'M')
index.to_period(freq='M')
```

**Transform the dtype**

```python
pd.to_datetime
pd.to_numeric
```


# Profile data
## Plotting
```python
from ggplot import *
ggplot(diamonds, aes(x='price', color='clarity')) + \
    geom_density() + \
    scale_color_brewer(type='div', palette=7) + \
    facet_wrap('cut')

ggplot(mtcars,aes(x="mpg",y="wt"))+\
    geom_point(color='red')
```

## Inspect samples of data

```python
dow.drop(['Open','High','Low','Adj Close','Volume'],axis=1,inplace=True)
dow.head()
```

## Numeric data
- Distribution

```python
data.select_dtypes('number').describe()

data.transactionAmount.hist(bins=20)

# construct normal distribution using mean and standard deviation of the data
from scipy.stats import norm
a = data3.values.flatten().mean()
b = data3.values.flatten().std()

x = np.linspace(-10,10,1000)
y = norm.pdf(x, loc=a, scale=b)

plt.figure(figsize=(16, 5))
plt.title('Distribution of stock returns vs normal distribution')
sns.distplot(data3.values.flatten(), hist=True, kde=True, 
             bins=40, 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
sns.lineplot(x,y, color="coral",label='Normal Distribution')
```
- Clustering

```python
from sklearn.cluster import KMeans
res = KMeans(n_clusters=2,random_state=1234).fit_predict(data2.T)
plt.figure()
plt.title('outliers')
data2.loc[:,res.astype(bool)].mean(axis=0).hist()
data2.loc[:,~res.astype(bool)].mean(axis=0).hist()
```

# Model data

1. Problem definition: regression or classification; evaluation metrics

2. Data exploration: understand the relationship between target variables and features using visualization and correlation analysis

3. Undersampling or oversampling 

3. Data cleaning: outliers and high-leverage points might have to be thrown out so as not to distort the model. 

```python
use_data = use_data.select_dtypes(['object','number'])
```

3. Feature selection and feature engineering: select features that are relevant to the target variable. We can also do some feature engineering, e.g. smoothing or clustering variables. 

```python
data.replace('',np.nan,inplace=True)
exclude_cols = data.columns[data.isnull().sum()==data.shape[0]].values
exclude_cols = np.append(exclude_cols,['cardLast4Digits','customerId',])
use_data = data.drop(exclude_cols,axis=1)
# replace all NAs with Other
use_data.fillna('Other',inplace=True)
```
4. Model training and evaluation: split the dataset. Train the candidate models on training set and test them on testing set, then select the model that performs the best. In the training process, we need to define the set of candidate models, loss function, hyperparameter space, optimizer and evaluation metrics. The candidates models and evaluation metrics would depend on the problem statement and data exploration. 

```python
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

models = []

models.append(('LR', LogisticRegression()))
models.append(('GBT', GradientBoostingClassifier(validation_fraction=0)))
models.append(('RF', RandomForestClassifier(n_estimators=100,min_samples_split=10)))

#testing models

results = {}
names = []

for name, model in models:
    kfold = KFold(n_splits=3, random_state=1)
    cv_results = cross_val_score(model, X_res, y_res, cv=kfold, scoring='f1')
    results[name]=cv_results
    print(name, ": {:.4f},{:.2f}".format(cv_results.mean(), cv_results.std()))

rf = models[2][1]
rf.set_params(kernel='linear').fit(X, y)  
rf.fit(X_res,y_res)

y_pred = rf.predict(X_test)
print(classification_report(y_test,y_pred))



from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

select = VarianceThreshold(0.04)
fit1 = RandomForestRegressor(random_state=0)

# construct pipeline object
pipeline = Pipeline(steps=[("select",select),('Regr',fit1)])

X_train, X_test, y_train, y_test = \
train_test_split(df, boston.target, test_size=0.33, random_state=42)


pipeline.fit(X_train,y_train)
print(pipeline.predict(X_test)[1:4])
print(fit1.fit(X_train,y_train).predict(X_test)[1:4])
report = skl.metrics.regression( y_test, y_predictions )
print(report)
```

5. Model diagnostics: investigate 1) whether the hyperparameters of the model make sense 2) whether feature importance/impact make sense 3) the source of prediction error: from bias or variance

6. Model enhancement: maybe new data(e.g. macroeconomic factors, company financials) can be brought in to enhance this model


