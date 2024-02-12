import json

# Define the word problems
multiplication_problems = [
    {"text": "Alice has 4 boxes, and each box contains 5 chocolates. How many chocolates does Alice have in total?", "label": 0},
    {"text": "There are 6 bags, and each bag has 8 candies. How many candies are there in total?", "label": 0},
    {"text": "A farmer has 5 fields, and each field has 7 cows. How many cows does the farmer have in total?", "label": 0},
    {"text": "If there are 4 shelves, and each shelf has 9 books, how many books are there in total?", "label": 0},
    {"text": "There are 3 boxes, and each box contains 6 pens. How many pens are there in total?", "label": 0}
]

derivative_problems = [
    {"text": "Find the derivative of f(x) = ln(x^2 + 2x + 3).", "label": 1},
    {"text": "Evaluate the integral of g(x) = cos(2x) from 0 to pi/2.", "label": 1},
    {"text": "Determine the absolute maximum and minimum values of the function h(x) = x^3 - 3x^2 + 2x - 1 on the interval [-1, 2].", "label": 1},
    {"text": "Calculate the area enclosed by the curve y = sin(x) and the x-axis on the interval [0, pi].", "label": 1},
    {"text": "Find the limit of f(x) as x approaches infinity, where f(x) = (2x^2 - x + 3) / (x^2 + 4).", "label": 1}
]


# Combine the problems
samples = multiplication_problems + derivative_problems

# Shuffle the samples (optional)
import random
random.shuffle(samples)

filename = "test_set.json"

# Write the samples to a JSON file
with open(filename, "w") as file:
    json.dump(samples, file, indent=4)