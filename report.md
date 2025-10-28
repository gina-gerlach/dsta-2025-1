# Milestone 1 - Report
Datascience Toolkits and Architectures
Gina Gerlach

## 1. Dataset Description

The MNIST handwritted digit database is an image collection of handwritten digits (numbers 0-9).

Size: 18.2MB
Number of rows: 70,000
Type: 28x28px black and white images
Source: Census Bureau employees and high school students (evenly distributed between testing and training sets)
Training set: 60,000 
Testing set: 10,000

Type of problem: Classification
Number of classes: 10 (1 per digit)
Images per class: 7,000 (6,000 training, 1,000 testing)

This database is used for creating pattern recognition methods with machine learning models.

## 2. Check out code base
Cloned to GitHub using curl (curl chosen because only needed the one document, not whole repository. Otherwise would have used git clone.)

Code:
curl -o mnist_convnet.py https://github.com/keras-team/keras-io/blob/master/examples/vision/mnist_convnet.py

## 3. Commit py file to Git Repo

Code:
git add mnist_convnet.py
git commit -m "Add initial MNIST code base and updated report" -m ""Action corresponds to milestone 1 task #3. Updates to report.md reflect steps taken to complete task #3."
