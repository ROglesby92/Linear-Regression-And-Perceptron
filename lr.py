#!/usr/bin/python
#
# Barebones Python implementation of Logistic Regression.
# Using Gradient Decent. ( NO NUMPY )
#
import sys
import re

from math import log
from math import exp
from math import sqrt

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
    data.append( (x,y) )
  return (data, varnames)

 
# Train a logistic regression model using batch gradient descent
def train_lr(data, eta, l2_reg_weight):
  numvars = len(data[0][0])
  w = [0.0] * numvars
  b = 0.0

  for e in range(MAX_ITERS):
    _weights = [0.0] * numvars
	# Update bias
    bsum = 0.0
    for x,y in data:
        z = y*(dot_product(w,x)+b)
        bi = sigmoid(-z)
        bi *= y
        bsum -= bi
        for xj in range(numvars):
            _weights[xj] -= (bi * x[xj])



    # Update weights and bias
    b = b - (eta * bsum )
    for i in range(numvars):
        w[i] = w[i] - eta * (_weights[i] + ((l2_reg_weight) * w[i]))

  return (w,b)

def dot_product(x,y):
    total = 0.0
    for i in range(len(x)):
        total += x[i] * y[i]
    return total


def sigmoid(z):
    try:
        return 1. / (1. + exp(-z))
    except OverflowError:
        return 0.0

# Predict the probability of the positive label (y=+1) given the
# attributes, x.
def predict_lr(model, x):
  (w,b) = model
  pred = 0.0
  for i in range(len(w)):
    pred += x[i] * w[i]

  return sigmoid(pred+b)


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
  if (len(argv) != 5):
    print('Usage: lr.py <train> <test> <eta> <lambda> <model>')
    sys.exit(2)
  (train, varnames) = read_data(argv[0])
  (test, testvarnames) = read_data(argv[1])
  eta = float(argv[2])
  lam = float(argv[3])
  modelfile = argv[4]

  # Train model
  (w,b) = train_lr(train, eta, lam)

  # Write model file
  f = open(modelfile, "w+")
  f.write('%f\n' % b)
  for i in range(len(w)):
    f.write('%s %f\n' % (varnames[i], w[i]))

  # Make predictions, compute accuracy
  correct = 0
  for (x,y) in test:
    prob = predict_lr( (w,b), x )
    #print(prob)
    if (prob - 0.5) * y > 0:
      correct += 1
  acc = float(correct)/len(test)
  print("Accuracy: ",acc)

if __name__ == "__main__":
  main(sys.argv[1:])
