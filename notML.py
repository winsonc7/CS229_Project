import json
import random

chem_words = ["mol", "gas", "temperature", "molar", "atomic", "water", "solution", "acid", "base", "compound", "reaction", "delta", "aqueous", "cell", "mass", "state", "atom", "precipitate", "energy", "pressure"]

phy_words = ["mole", "gas", "volume", "pressure", "surface", "distance", "speed", "velocity", "density", "particle", "temperature", "current", "time", "light", "energy", "force", "delta", "wavelength", "acceleration", "vec"]

math_words = ["equal", "continuous", "circle", "radius", "root", "minimum", "maximum", "sin", "theta", "angle", "number", "axis", "point", "integer", "square", "log", "function", "cos", "tan", "positive"]

def convert_to_list(dataset_path):
    with open(dataset_path, 'r') as f:
        data = json.load(f)
        return data
    
def accuracy(truth, pred):
    score = 0
    for i in range(len(truth)):
        if truth[i] == pred[i]:
            score += 1
    return score / len(truth)

def run_model(data):
    subjects = ["chem", "phy", "math"]
    truth = []
    prediction = []
    for i in range(len(data)):
        # chem count, phy count, math count
        labels = [0, 0, 0]
        for word in chem_words:
            labels[0] += data[i]["question"].count(word)
        for word in phy_words:
            labels[1] += data[i]["question"].count(word)
        for word in math_words:
            labels[1] += data[i]["question"].count(word)
        if labels.count(max(labels)) != 3:
            prediction.append(labels.index(max(labels)))
        else:
            prediction.append(random.randint(0,2))
        truth.append(subjects.index(data[i]["subject"]))
    print(accuracy(truth, prediction))

train_path = "stem_data/stem_train.json"
test_path = "stem_data/stem_test.json"

train_data = convert_to_list(train_path)
test_data = convert_to_list(test_path)

run_model(train_data)
run_model(test_data)