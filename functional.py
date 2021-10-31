"""
Return the evaluation of different datasets given one model.
"""

from evaluate import evaluate_cls

report_col = [
    'method',
    'dataset',
    'pr_m',
    'rec_m',
    'f1_m',
    'pr_w',
    'rec_w',
    'f1_w',
    'auc',
    'pr']


def res_to_dict(method, dataset, res, report_col=report_col):
    """
    Return results as a dictionnary.
    Args:
        method (str): name of model
        dataset (str): name of dataset
        res (list): value list of metrics results to map into dictionnary
        report_col (list): key list of information to map into dictionnary
    """
    res.insert(0, dataset)
    res.insert(0, method)
    return {k: v for k, v in zip(report_col, res)}


def add_cls_res(cls_name, cls, X_train_sc, y_train, X_test_sc, y_test,
                X_train_os, y_train_os, X_train_sc_pre,
                X_test_sc_pre, X_train_pca, X_test_pca):
    """
    cls_name (str): name of model
    X_train_sc (array): all features training dataset 
    ...
    X_test_pca (array): PCA embedded testng dataset.
    """
    results_df = []

    # Save results for all dataset
    print('--> All')
    cls.fit(X_train_sc, y_train)
    results_df.append(
        res_to_dict(
            cls_name,
            'all',
            evaluate_cls(
                cls,
                X_train_sc,
                y_train,
                X_test_sc,
                y_test,
                'minimal')))

    # Save result for OS dataset
    print('--> Oversample')
    cls.fit(X_train_os, y_train_os)
    results_df.append(
        res_to_dict(
            cls_name,
            'os',
            evaluate_cls(
                cls,
                X_train_os,
                y_train_os,
                X_test_sc,
                y_test,
                'minimal')))

    # Save result for preselect dataset
    print('--> Preselected')
    cls.fit(X_train_sc_pre, y_train)
    results_df.append(
        res_to_dict(
            cls_name,
            'pre',
            evaluate_cls(
                cls,
                X_train_sc_pre,
                y_train,
                X_test_sc_pre,
                y_test,
                'minimal')))

    # Save result for PCA dataset
    print('--> PCA')
    cls.fit(X_train_pca, y_train)
    results_df.append(
        res_to_dict(
            cls_name,
            'pca',
            evaluate_cls(
                cls,
                X_train_pca,
                y_train,
                X_test_pca,
                y_test,
                'minimal')))
    return results_df
