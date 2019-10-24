[https://mode.com/blog/python-data-cleaning-libraries?source=post_page-----af1edfbe2a3----------------------](useful packages)

# Read data

1. What are the columns and types? Change dtype if necessary

```python
df.info()
data = pd.ExcelFile('reshaping_data.xlsx')
data.sheet_names
data.parse(sheetname='ABC_inc', skiprows=0).head(10)

data = pd.read_csv(‘movie_metadata.csv’, 
	dtype={‘duration’: int, "title_year": str})
```

2. Indexing?

```python
df.index.values
df.set_index('column_name_to_use', inplace=True)
```



# Clean data
## Drop unwanted columns
```python
df.drop(columns_to_drop, inplace=True, axis=1)
```
## Outliers

```python
# remove all rows with outliers in at least one row
df = df[(np.abs(stats.zscore(df.drop(['DATE'], axis=1))) < 3).all(axis=1)]
# show final size after removing outliers
df.shape
```
## Uniqueness

```python
df.iloc[:,0].is_unique
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

```python
df['ST_NUM'].fillna(125, inplace=True)
# limit to fill only one value
df.fillna(method='pad/ffill/bfill', limit=1)
# drop columns which have at least 90% non-NaNs
df.dropna(thresh=int(df.shape[0] * .9), axis=1)
df.dropna(axis=1)
data.dropna(subset=[‘title_year’])

```


# Process data
## Encode

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
import sklearn.preprocessing as prep
lb = prep.LabelEncoder()
res = lb.fit_transform(df.iloc[:,0])
df.apply(lb.fit_transform)
# one-hot encoder in pandas
pd.get_dummies(df)

```
## Normalize
Sklearn scaler

```python
from sklearn.preprocessing import MinMaxScaler, Binarizer
mm_scaler = MinMaxScaler(feature_range=(0, 1)) # scale to 0 and 1
tz = Binarizer(threshold=-25.0).fit([ary_int])

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
**Pivoting**

```python
# all the identifier columns to be included in the multi-index 
idx =['district','province','partner','financing_source'
,'main_organization']
pd.melt(df, id_vars=idx, value_vars=['B', 'C'])
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

**Transform the dtype**

```python
pd.to_datetime
pd.to_numeric
```


# Profile data
## Inspect samples of data

```python
dow.drop(['Open','High','Low','Adj Close','Volume'],axis=1,inplace=True)
dow.head()
```

## Numeric data
- Distribution



# Model data



Handling missing values
* Binary features: -1 for negative, 0 for missing, 1 for positive
* Numeric features:
- Tree-based methods: encode as a big positive or negative number, 2*max(x), 2*min(x), -99999
- Linear, neural nets: encode by splitting into 2 columns: 1. Binary column indicating NA(0 if not and 1 if yes) 2. Original column replace NAs with mean or median
* Categorical features:
- Encode as an unique category
- Replace with the most frequent category level

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

Feature interaction
- Analyze GBM splits linear regression weights

Target Transformation
- log(x), log(x+1), sqrt(x), sqrt(x+1) etc
