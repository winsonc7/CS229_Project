from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
import json
import re
import numpy as np
import csv

VECSIZE = 20

def clean_text(text):
  return re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())

class_dict = ["math", "chem", "phy"]

def main(data_file):
  with open(data_file, 'r') as f:
    data = json.load(f)

  question = [item['question'] for item in data]
  for i in range(len(question)):
      question[i] = sent_tokenize((question[i]))
      question[i] = word_tokenize(clean_text(question[i][0]))
  model = Word2Vec(sentences=question, vector_size=VECSIZE, window=3, min_count=4, workers=4)
  sentence_embeddings = []
  for i in range(len(question)):
      vec_sum = np.zeros(model.vector_size)
      count = 0
      for token in question[i]:
          if token in model.wv:
              vec_sum += model.wv[token]
              count += 1
      if count > 0:
          new = vec_sum / count
          sentence_embeddings.append(np.concatenate(([class_dict.index(data[i]['subject'])], new)))
      else:
          # Handle cases where all tokens are out-of-vocabulary
          sentence_embeddings.append(np.zeros(model.vector_size + 1))

  columns = ['y'] + ['x' + str(i) for i in range(sentence_embeddings[0].shape[0])]
  csv_file = f"{data_file[:-5]}_word2vec_{VECSIZE}.csv"
  with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    # Write column headers
    writer.writerow(columns)
    # Write data rows
    for sample in sentence_embeddings:
      # Write row to CSV file
      writer.writerow(sample)

data_file2 = "stem_data/stem_test.json"
data_file1 = "stem_data/stem_train.json"
main(data_file1)
main(data_file2)