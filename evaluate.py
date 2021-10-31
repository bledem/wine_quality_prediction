"""
Functions to evaluate one model with different metrics.
#TODO transform these functions into class.
"""

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import plot_confusion_matrix, roc_auc_score, classification_report, auc, average_precision_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, precision_recall_curve

from vars import colors


def evaluate_cls(cls, X_train, y_train, X_test, y_test, verbose='maximal'):
    """
    Args:
        cls (scikit-learn model)
        X_train (array)
        y_train (array)
        X_test (array)
        y_test (array)
        verbose (str) ``minimal`` or ``maximal``.
    Output:
        List of metrics
    """
    if len(np.unique(y_train)) > 6:
        ovr = True
    else:
        ovr = False
    print(f'Training acc: {cls.score(X_train, y_train):.2f}')
    print(f'Test acc: {cls.score(X_test, y_test):.2f}')
    try:
        # Calculate roc_auc score with multiclass parameter
        if ovr:
            roc_auc = roc_auc_score(
                y_test,
                cls.predict_proba(X_test),
                multi_class='ovr')
        else:
            roc_auc = roc_auc_score(y_test, cls.predict(X_test))
        print(f'Roc AUC score {roc_auc:.2f}')
        auc_pr = []
        pred_prob = cls.predict_proba(X_test)
        labels = np.unique(y_test)
        for i in range(len(labels)):
            precision, recall, _ = precision_recall_curve(
                y_test, pred_prob[:, i], pos_label=labels[i])
            auc_pr.append(auc(recall, precision))
        auc_pr = np.mean(auc_pr)
        print(f'Precision-Recall AUC score {auc_pr:.2f}')
    except BaseException:
        auc_pr = None
        roc_auc = None
        print('AUC non available')
    # Precision, recall, f1 score by class
    test_pred = cls.predict(X_test)
    if len(np.unique(test_pred)) > 8:
        test_pred = np.rint(cls.predict(X_test))
    print(classification_report(y_test, test_pred, zero_division=1))
    out_dict = classification_report(
        y_test, test_pred, zero_division=1, output_dict=True)
    m_dict, w_dict = out_dict['macro avg'], out_dict['weighted avg']
    pr_mac, rec_mac, f1_mac = m_dict['precision'], m_dict['recall'], m_dict['f1-score']
    pr_w, rec_w, f1_w = w_dict['precision'], w_dict['recall'], w_dict['f1-score']
    # CV accuracy and CM
    if verbose == 'maximal':
        # Cross validation
        accuracy = cross_val_score(
            cls, X_train, y_train, scoring='accuracy', cv=5)
        print(f'Cross validation score with roc_auc {accuracy.mean():.2f}')
        # Confusion matrix
        print('--Confusion Matrix--')
        plot_confusion_matrix(cls, X_test, y_test)
    return [pr_mac, rec_mac, f1_mac, pr_w, rec_w, f1_w, roc_auc, auc_pr]


def plot_roc_pr_curves(clf, X_test, y_test):
    """
    Plot ROC and Precision-Recall curves.
    Args:
        clf (scikit-learn model)
        X_test (array)
        y_test (array)
    """
    # fp / (tp+fp)
    fpr = {}
    # tpr = recall
    tpr = {}
    auc_pr = []
    thresh = {}

    _, axs = plt.subplots(2, 1, figsize=(10, 10))
    pred_prob = clf.predict_proba(X_test)
    labels = np.unique(y_test)

    for i in range(len(labels)):
        fpr[i], tpr[i], thresh[i] = roc_curve(
            y_test, pred_prob[:, i], pos_label=labels[i])
        precision, recall, thresholds = precision_recall_curve(
            y_test, pred_prob[:, i], pos_label=labels[i])
        axs[0].plot(
            fpr[i],
            tpr[i],
            linestyle='--',
            color=colors[i],
            label=f'Class {labels[i]} vs Rest')
        axs[1].plot(
            recall,
            precision,
            linestyle='--',
            color=colors[i],
            label=f'Class {labels[i]} vs Rest')
        auc_pr.append(auc(recall, precision))

    if len(labels) > 2:
        roc_auc = roc_auc_score(
            y_test,
            clf.predict_proba(X_test),
            multi_class='ovr')
    else:
        roc_auc = roc_auc_score(
            y_test, clf.predict_proba(X_test)[
                :, 1], labels=labels)

    axs[0].plot([0, 1], [0, 1], label='random guessing')
    axs[0].set_title(f'Multiclass ROC curve AUC:{roc_auc:.2f}')
    axs[0].set_xlabel('False Positive Rate')
    axs[0].set_ylabel('True Positive rate')
    axs[0].legend(loc='best')

    axs[1].set_title(
        f'Multiclass Precision-Recall curve {np.mean(auc_pr):.2f}')
    # axs[1].plot([1,0], [0,1], label='random guessing')

    axs[1].set_xlabel('Recall')
    axs[1].set_ylabel('Precision')
    axs[1].legend(loc='best')
    plt.tight_layout()
