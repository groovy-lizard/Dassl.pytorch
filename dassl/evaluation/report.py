"""Report module"""
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import glob


def list_item_swap(item_list, i1, i2):
    """Swap item_list items given value

    :param item_list: item_list containing the items
    :type item_list: list
    :param i1: value of item 1
    :type i1: any
    :param i2: value of item 2
    :type i2: any
    :return: swapped list
    :rtype: list
    """
    i = item_list.index(i1)
    j = item_list.index(i2)
    item_list[i], item_list[j] = item_list[j], item_list[i]
    return item_list


def get_col_list(df, metric_name):
    """Get the list of columns from a given dataframe

    :param df: dataframe with predictions
    :type df: pd.DataFrame
    :param metric_name: name of the metric to be used
    :type metric_name: str
    :return: list of columns from given df
    :rtype: list
    """
    if metric_name == "balanced_accuracy":
        col_list = list(df.drop(
            columns=['file', 'race_preds', 'race']).keys())
    else:
        col_list = list(df.drop(columns=['file', 'race_preds']).keys())
        col_list = list_item_swap(col_list, 'age', 'gender')
    col_list = list_item_swap(col_list, 'age', 'gender')
    return col_list


def fix_age_order(age_list):
    """Correctly sort ages list

    :param age_list: list of ages from df uniques
    :type age_list: list
    :return: new sorted age list
    :rtype: list
    """
    age_list[0] = '03-09'
    age_list[-1] = '00-02'
    age_list.sort()
    return age_list


def get_uniques(df, col_list):
    """Get Unique values from columns

    :param df: dataframe with predictions
    :type df: pd.DataFrame
    :param col_list: list of columns from dataframe
    :type col_list: list
    :return: list of unique values from column
    :rtype: list
    """
    uniques = []
    for col in col_list:
        col_items = list(df[col].unique())
        if col == "age":
            col_items = fix_age_order(col_items)
        for unique in col_items:
            if col == "age" and unique == "00-02":
                unique = "0-2"
            elif col == "age" and unique == "03-09":
                unique = "3-9"
            uniques.append(unique)
    return uniques


def get_empty_report_dict(df, metric_name):
    """Generate a new empty report dictionary

    :param df: dataframe with predictions to extract columns
    :type df: pd.DataFrame
    :param metric_name: name of the metric to be used
    :type metric_name: str
    :return: an empty report dictionary
    :rtype: dict
    """
    report_dict = {
        'Mode': [],
        metric_name: [],
    }

    col_list = get_col_list(df, metric_name)
    uniques = get_uniques(df, col_list)
    for unique in uniques:
        report_dict[unique] = []
    return report_dict


def metric_loader(metric_name):
    """Loads the corresponding metric function"""
    metric_functions = {
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score
    }
    return metric_functions[metric_name]


def filter_df(df, col, val):
    """Filter dataframe by column and value"""
    return df[df[col] == val]


def accuracy_eval(df, metric):
    """Return the accuracy score of the true label vs predictions"""
    return round(metric(df['race'], df['race_preds']), 4)


def gen_dict_report(df, mode_name, metric_name, rep_dict):
    """Generate the dictionary report

    :param df: dataframe with predictions
    :type df: pd.DataFrame
    :param mode_name: name of the prediction mode used
    :type mode_name: str
    :param metric_name: name of the metric used
    :type metric_name: str
    :param rep_dict: current report dictionary
    :type rep_dict: str
    :return: updated report dictionary
    :rtype: dict
    """
    metric_func = metric_loader(metric_name)
    rep_dict['Mode'].append(mode_name)
    rep_dict[metric_name].append(accuracy_eval(df, metric_func))
    col_list = get_col_list(df, metric_name)
    for col in col_list:
        uniques = get_uniques(df, [col])
        for unique in uniques:
            col_df = filter_df(df, col, unique)
            col_acc = accuracy_eval(col_df, metric_func)
            rep_dict[unique].append(col_acc)
    return rep_dict


def fface_report(preds_df):
    """Generate final FairFace report"""
    metric_name = "accuracy"
    rep_dict = get_empty_report_dict(preds_df, metric_name)
    rep_dict = gen_dict_report(
                preds_df, "CoOp", metric_name, rep_dict)
    return pd.DataFrame(rep_dict)
