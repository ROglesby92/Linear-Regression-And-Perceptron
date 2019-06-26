#!/usr/bin/python
#
# Perceptron Implementation (NO NUMPY)
#
# Author: Ryan Oglesby
#
#
import sys
import re
from math import log
from math import exp

MAX_ITERS = 100

# Load data from a file
def read_data(filename):
  f = open(filename, 'r')
  p = re.compile(',')
  data = []
  header = f.readline().strip()
  varnames = p.split(header)
  namehash = {}
  for l in f:
    example = [int(x) for x in p.split(l.strip())]
    x = example[0:-1]
    y = example[-1]
    # Each example is a tuple containing both x (vector) and y (int)
    data.append( (x,y) )
  return (data, varnames)


# Learn weights using the perceptron algorithm
def train_perceptron(data):
  # Initialize weight vector and bias
  numvars = len(data[0][0])
  w = [0.0] * numvars
  b = 0.0
  for i in range(MAX_ITERS):
	for x,y in data:
		model = (w,b)
		pred = predict_perceptron(model,x)
		if pred * y <= 0:
			for j in range(len(w)):
				w[j] = w[j] + y*x[j]
			b = b + y
				

  return (w,b)


# Compute the activation for input x.
# (NOTE: This should be a real-valued number, not simply +1/-1.)
def predict_perceptron(model, x):
  (w,b) = model
  activation = 0.0;
  for i in range(len(x)):
	activation += w[i] * x[i]

  return activation + b


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
  # Process command line arguments.
  if (len(argv) != 3):
    print('Usage: perceptron.py <train> <test> <model>')
    sys.exit(2)
  (train, varnames) = read_data(argv[0])
  (test, testvarnames) = read_data(argv[1])
  modelfile = argv[2]

  # Train model
  (w,b) = train_perceptron(train)

  # Write model file
  f = open(modelfile, "w+")
  f.write('%f\n' % b)
  for i in range(len(w)):
    f.write('%s %f\n' % (varnames[i], w[i]))

  # Make predictions, compute accuracy
  correct = 0
  for (x,y) in test:
    activation = predict_perceptron( (w,b), x )
    #print(activation)
    if activation * y > 0:
      correct += 1
  acc = float(correct)/len(test)
  print("Accuracy: ",acc)

if __name__ == "__main__":
  main(sys.argv[1:])
