from get_csv import get_csv_data
import pandas as pd


def merge(csv1, csv2, csv3):
    with open(csv1, 'r') as fa:  # 读取需要拼接的前面那个TXT
        with open(csv2, 'r') as fb:  # 读取需要拼接的后面那个TXT
            with open(csv3, 'w') as fc:  # 写入新的TXT
                for line in fa:
                    fc.write(line)
                for line in fb:
                    fc.write(line)


def add_list(file, name):
    """
    :param file: csv
    :param name: label
    add an abel in the first of csv
    """
    csv = pd.read_csv(file, header=None)
    shwj = [name] * len(csv)
    csv.insert(0, "shwj", shwj, allow_duplicates=False)
    csv.to_csv("data/" + name + ".csv", index=False, header=None)
    # csv.to_csv(name+"ceshi.csv",index=False,header=None)


def joint_data(path):
    get_csv_data(path + "up", name="up")
    get_csv_data(path + "fall", name="fall")

    add_list("data/up.csv", "up")
    add_list("data/fall.csv", "fall")
    # df1 = pd.read_csv("data/up.csv")
    # df2 = pd.read_csv("data/fall.csv")

    merge("data/up.csv","data/fall.csv","data/fall_vs_up.csv")


if __name__ == '__main__':
    joint_data("F:\\programs\\data\\IMG\\falling_ustb\\")
