import numpy as np
import pandas as pd
import statistics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# converting uploaded file to dataframe object
def load_data(uploaded_file):    
    ''' 
    Argument
    uploaded_file: Takes uploaded file object
    ---------------------
    Return 
    If file object is not None then returns a dataframe object
    '''
    if uploaded_file is not None:
        # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(uploaded_file, sep=',')
        # st.write(dataframe)
        # print(dataframe.shape)
        return dataframe

def get_n_rows(df : pd.DataFrame,n=5,from_start=True) -> pd.DataFrame:
    '''
    Argument
    df: dataframe
    n: int, default: 5 | number of rows
    from_start: boolean, default: True | If True returns first n samples otherwise returns last n samples. 
    ---------------------
    Return
    Dataframe contains first n samples
    '''
    if(from_start):
        return df.head(n)
    return df.tail(n)

def get_target(df : pd.DataFrame , col_name : str)->pd.Series:
    '''
    Arguments
    col_name: column name, which user wants to treat as target
    ---------------------------------
    Return
    The target column from the data and also shows that which type of problem(e.g, classification, 
    regression) it can we after chossing the target.  
    '''
    return df[col_name]

def get_info(df : pd.DataFrame)-> None:
    '''
    Argument
    df : dataframe object
    ---------------------------------
    Return
    print the info dataframe which contains the name and data type of every column
    '''
    cols=df.dtypes.keys().tolist()
    info_df=pd.DataFrame({'Columns': cols})
    info_df['Data Type']=info_df['Columns'].apply(lambda x: df[x].dtype)
    info_df['Num of Unique Instances']=df.nunique().values.tolist()
    return info_df

# drop duplicate
# df.drop_duplicates(inplace=True)

def get_null_info(df : pd.DataFrame)-> None:
    '''
    Argument
    df : dataframe object
    ---------------------------------
    Return
    print the null info dataframe which contains the name, count of null values and the percentage of
    and data type of every column
    '''
    null_info_df=pd.DataFrame({'Columns': df.columns})
    null_info_df['Null Count']=null_info_df['Columns'].apply(lambda x: df[x].isna().sum())
    null_info_df['Null Percentage']=null_info_df['Columns'].apply(lambda x: df[x].isna().sum()/len(df))
    return null_info_df

def drop_column(df : pd.DataFrame , col_name : str)-> None:
    '''
    Argument
    df : dataframe object
    col_name : String | Name of the column which user wants to drop
    --------------------------------
    Return
    None
    '''
    df.drop([col_name],axis=1,inplace=True)
    

def handle_missing(df: pd.DataFrame,col_name: str,method: str)->None: 
    '''
    Argument
    df: dataframe object
    col_name: String | column name on which we wants to handle missing value.
    method: String | drop, Mean, Median, Mode, Min, Max|the method users wants to use for filling the missin values.
    -------------------------------------
    Return
    None
    '''
    if(method=='drop'):
        drop_column(df,col_name)
    elif(method=='mean'):
        df[col_name] = df[col_name].replace(np.NaN, df[col_name].mean())
    elif(method=='median'):
        df[col_name] = df[col_name].replace(np.NaN, df[col_name].median())
    elif(method=='mode'):
        df[col_name] = df[col_name].replace(np.NaN, statistics.mode(df[col_name]))                
    elif(method=='max'):
        df[col_name] = df[col_name].replace(np.NaN, df[col_name].max())        
    elif(method=='min'):
        df[col_name] = df[col_name].replace(np.NaN, df[col_name].min())        
    else:
        print("Method is Not Recognized")

def show_missing(df: pd.DataFrame)-> None:
    '''
    Argument
    df: dataframe object
    Shows the barplot for the count of null values corresponding to each feature.
    ---------------------------
    Return
    None'''
    cols=df.isna().sum().keys()
    null_count=df.isna().sum().values
    sns.barplot(cols,null_count)
    plt.xticks(rotation=90)
    plt.show()  


def get_stats(df: pd.DataFrame, type : str)->None:
    '''
    Argument
    df: dataframe object
    type: String | 'number' , 'object' , 'all' | shows stats related to numeric, categorical or all the 
    features.
    -----------------
    Return
    None
    '''
    print(df.describe(include=type).T)


def encode(df : pd.DataFrame, type = 'label')-> pd.DataFrame:
    '''
    df : dataframe object
    type : String | 'label' , 'One-hot' | Type of Encoding 
    --------------------------------------
    Return 
    It will returns new dataframe with encoded features.
    '''
    cat_features=cat_features=[col for col in df.columns if df[col].dtype=='O'] 
    if(type=='label'):
        for col_name in cat_features:
            encoder=LabelEncoder()
            df[col_name]=encoder.fit_transform(df[col_name])
    else:
        df=pd.get_dummies(df,drop_first=True)
    return df

def plot_correlation(df : pd.DataFrame ,method : 'pearson') -> None :
    '''
    df : dataframe object
    method : String | default : 'pearson' | 'pearson', 'kendall', 'spearman | method which user wants to 
    use for measuring the correlation.
    --------------------------------------
    Return 
    It will display the heatmap which shows the correlation between different features in the dataframe.
    '''
    plt.figure(figsize=(6,6))
    plt.title('Correlation Map')
    sns.heatmap(df.corr(method=method),
                annot=True)


def plot_triu(df : pd.DataFrame , method='pearson') -> None :
    '''
    df : dataframe object
    method : String | default : 'pearson' | 'pearson', 'kendall', 'spearman | method which user wants to 
    use for measuring the correlation.
    --------------------------------------
    Return 
    It will display the left triangle heatmap which shows the correlation between different features in 
    the dataframe.
    '''
    mask = np.triu(np.ones_like(df.corr()))
    plt.figure(figsize=(6,6))
    plt.title('Correlation Map')
    ax=sns.heatmap(df.corr(method=method),
                annot=True,  mask=mask)