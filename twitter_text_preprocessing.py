import sys, os, re
import pandas as pd
import csv

punctuation_str = '''!()-[]`{};:\,<>./?@#$%^&+|*_~'''
punctuation_str_2 = '''\'\"'''
table = str.maketrans(dict.fromkeys(punctuation_str, " "))
table_2 = str.maketrans(dict.fromkeys(punctuation_str_2, ""))

def read_csv_file(file_name, data_col, label_col):
    data_list = []
    label_list = []
    with open(file_name, 'r') as csv_reader:
        reader = csv.reader(csv_reader, delimiter=',')
        for line_number, row in enumerate(reader):
            if line_number!=0:
                data_list.append(row[data_col])
                label_list.append(row[label_col])

    return data_list, label_list

def clean_text(input_string):
    
    input_string = input_string.split('\n')
    input_string = " ".join(input_string)
    input_string = re.sub(r'\(\(\(.*\)\)\)','bad jew conspiracy',input_string)
    input_string = re.sub(r'@\S+', " usermention ", input_string)
    input_string = re.sub(r'#'," ", input_string)
    input_string = re.sub(r'\d\S+'," ", input_string)
    input_string = re.sub(r"http\S+", " ", input_string)
    input_string = re.sub(r'www\S+', " ", input_string)
    input_string = re.sub(r'\.|/|:|-', " ", input_string)
    input_string = re.sub(r'[^\w\s]','',input_string)
    input_string = " ".join(input_string.split())

    return input_string


def parse_input_character(data):
    train_data = []
    train_label = []
    for row in data:
        temp_row  = row.split()
        char_list = []
        for words in temp_row:
            for ch in words:
                char_list.append(ch)
        train_data.append(char_list)

    return train_data



def prepare_dataset(filename, data_col, label_col, network_type):

    print("########################-Using Twitter Based Pre-processing-##################")

    original_data, original_labels = read_csv_file(filename, data_col, label_col)
    for i in range(len(original_data)):
        original_data[i] = clean_text(original_data[i])

    if network_type=="character-based":
        train_data = parse_input_character(original_data)
        return train_data, original_labels
    else:
        return original_data, original_labels