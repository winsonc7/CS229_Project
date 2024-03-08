import json


#This file defines a class that is meant to read in the JEEBench data into more easily handled formats like lists
def main():
    #Example usage of class
    filename = "JEEBench_data/dataset.json"
    data_reader = JEEBench_reader(filename)
    data_reader.read_data()
    data_reader.read_data_to_list()

    print(len(data_reader.problem_list))
    print(data_reader.problem_list[50])

    # print(type(data_reader.data))
    # print(data_reader.data[0])


class JEEBench_reader:
    def __init__(self, filename):
        self.filename = filename
        self.problem_list = []   #list where each element is a string with problem text
        self.prob_list_w_labels = [] #list where each element is a tuple :  (problem text, subject label)
        self.prob_labels = [] #Problem labels using strings
        self.prob_labels_numbered = [] #Problem labels but using numbers. Number scheme is : phy = 0, chem = 1, math = 2

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

if __name__ == '__main__':
    main()

