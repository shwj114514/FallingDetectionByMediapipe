from get_csv import get_csv_data
import pandas as pd


def add_list(file, name):
    """
    :param file: csv
    :param name: label
    add an abel in the first of csv
    """
    csv = pd.read_csv(file, header=None)
    shwj = [name] * len(csv)
    csv.insert(0, "shwj", shwj, allow_duplicates=False)
    csv.to_csv(name + ".csv", index=False, header=None)
    # csv.to_csv(name+"ceshi.csv",index=False,header=None)


def joint_data(path):
    get_csv_data(path+"up", name="up")
    get_csv_data(path+"fall", name="fall")

    add_list("up.csv", "up")
    add_list("fall.csv", "fall")
    df1 = pd.read_csv("up.csv")
    df2 = pd.read_csv("fall.csv")

    # merge
    df = pd.concat([df1,df2])
    # remove duplicate data
    df.drop_duplicates()
    df.to_csv('fall_vs_up.csv', encoding='utf-8')


if __name__ == '__main__':
    joint_data("F:\\programs\\data\\IMG\\falling_ustb")
