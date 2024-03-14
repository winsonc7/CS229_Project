import json
import numpy as np
import random

#This file defines a class that is meant to read in the JEEBench data into more easily handled formats like lists
def main():
    #Example usage of class
    filename = "JEEBench_data/dataset.json"
    train_split = .8
    data_reader = JEEBench_reader(filename)
    data_reader.read_data()
    data_reader.read_data_to_list()

    print(len(data_reader.problem_list))
    #print(data_reader.problem_list[50])

    train, test, train_labels, test_labels = data_reader.split_train_test(train_percent=train_split, verbose=True)

    print(train[1])

class JEEBench_reader:
    def __init__(self, filename):
        self.filename = filename
        self.problem_list = []   #list where each element is a string with problem text
        self.prob_list_w_labels = [] #list where each element is a tuple :  (problem text, subject label)
        self.prob_labels = [] #Problem labels using strings
        self.prob_labels_numbered = [] #Problem labels but using numbers. Number scheme is : phy = 0, chem = 1, math = 2
        self.train = []
        self.test = []

    def read_data(self):
        with open(self.filename, 'r') as json_file:
            self.data = json.load(json_file)

    def read_data_to_list(self):
        for point in self.data:
            self.problem_list.append(point["question"])
            self.prob_list_w_labels.append((point["question"], point["subject"]))
            self.prob_labels.append(point["subject"])
            if point["subject"] == "phy":
                self.prob_labels_numbered.append(0)
            elif point["subject"] == "chem":
                self.prob_labels_numbered.append(1)
            else:
                self.prob_labels_numbered.append(2)

    def split_train_test(self, train_percent=.8, verbose=False):
        data = self.prob_list_w_labels
        chem_data = []
        math_data = []
        phys_data = []
        #print("data : ", data[0])
        for i in range(len(data)):
            if data[i][1] == "phy":
                phys_data.append(data[i])
            elif data[i][1] == "chem":
                chem_data.append(data[i])
            else:
                math_data.append(data[i])

        num_train_chem = int(len(chem_data) * train_percent)
        num_train_math = int(len(math_data) * train_percent)
        num_train_phys = int(len(phys_data) * train_percent)

        train = chem_data[:num_train_chem] + math_data[:num_train_math] + phys_data[:num_train_phys]
        test = chem_data[num_train_chem:] + math_data[num_train_math:] + phys_data[num_train_phys:]

        random.shuffle(train)
        random.shuffle(test)

        self.train = train
        self.test = test

        if (verbose):
            print("Total Number of training ex : ", len(train))
            print("Total Number of testing ex : ", len(test))
            print("Number of chem training ex : ", num_train_chem)
            print("Number of physics training ex : ", num_train_phys)
            print("Number of math training ex : ", num_train_math)
            print("Number of chem testing ex : ", len(chem_data) - num_train_chem)
            print("Number of physics testing ex : ", len(phys_data) - num_train_phys)
            print("Number of math testing ex : ", len(math_data) - num_train_math)

        train_labels = self.get_labels(train)
        test_labels = self.get_labels(test)
        train_no_labels = self.remove_labels(train)
        test_no_labels = self.remove_labels(test)
        return train_no_labels, test_no_labels, train_labels, test_labels

    def get_labels(self, data):
        result = []
        for i in range(len(data)):
            if data[i][1] == "phy":
                result.append(0)
            elif data[i][1] == "chem":
                result.append(1)
            else:
                result.append(2)
        return result

    def remove_labels(self, data):
        result = []
        for i in range(len(data)):
            result.append(data[i][0])
        return result



if __name__ == '__main__':
    main()

