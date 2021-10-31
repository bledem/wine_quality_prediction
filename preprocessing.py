"""
Function to pre-process the data before training.
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def remove_outlier(df_in, col_name):
    """
    Remove outliers from dataset.
    Args:
        df_in (dataframe):
        col_name (str): name of the column(s) we remove the outliers
    Return:
        df_out (dataframe): dataframe without outliers.
    """
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3 - q1  # Interquartile range
    fence_low = q1 - 1.5 * iqr
    fence_high = q3 + 1.5 * iqr
    index_set = []
    for c in col_name:
        index_set.extend(df_in.loc[(df_in[col_name][c] < fence_low[c]) | (
            df_in[col_name][c] > fence_high[c])].index.values)
    index_set = list(set(index_set))
    df_out = df_in[~df_in.index.isin(index_set)]
    return df_out


def create_sets(wine_df, feat_col, tgt_col, RANDOM_STATE=42):
    """
    Generate train, val, test scaled dataset.
    Args:
        wine_df (dataframe): input dataframe
        feat_col (str): feature(s) column name
        tgt_col (str): target columns name
        random_state (int)
    Return: 
        X_train_sc (array)
        ...
        y_test (array)
        scaler (scikit scaler)
    """
    # Creating test set
    # Stratify guarantees we have same proportion in train and test
    # than original dataset
    X_train, X_test, y_train, y_test = train_test_split(wine_df[feat_col].values, wine_df[tgt_col].values,
                                                        test_size=0.15,
                                                        random_state=RANDOM_STATE,
                                                        stratify=wine_df[tgt_col].values)
    y_train, y_test = y_train.ravel(), y_test.ravel()

    # Creating validation set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.15,
                                                      random_state=RANDOM_STATE,
                                                      stratify=y_train)

    print(
        f'Train shape: {X_train.shape}, validation shape: {X_val.shape}, test shape: {X_test.shape}')

    # Most of the machine learning models needs to standardize the input
    # not to be biased by the amplitude of the values.

    scaler = StandardScaler()

    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    X_val_sc = scaler.transform(X_val)

    return X_train_sc, X_val_sc, X_test_sc, y_train, y_val, y_test, scaler
