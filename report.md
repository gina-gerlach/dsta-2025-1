# Milestone 1 - Report
<p>Datascience Toolkits and Architectures<br>
Gina Gerlach<br>

### Base set up:
- Created an Ubuntu 22.04 (ARM64) virtual machine using UTM on macOS.

- Allocated 4 GB RAM and set up a shared folder for file transfer between macOS and Ubuntu.

- Installed Python, Git, and other necessary tools<p>

> ```sudo apt install -y python3 python3-pip git python3-venv wget unzip build-essential curl```


## 1. Dataset Description
<p>The MNIST handwritted digit database is an image collection of handwritten digits (numbers 0-9).<br>

- Size: 18.2MB

- Number of rows: 70,000

- Type: 28x28px black and white images

- Source: Census Bureau employees and high school students (evenly distributed between testing and training sets)

- Training set: 60,000 

- Testing set: 10,000


- Type of problem: Classification

- Number of classes: 10 (1 per digit)

- Images per class: 7,000 (6,000 training, 1,000 testing) <br>
This database is used for creating pattern recognition methods with machine learning models.<p>

## 2. Check out code base
<p>Viewed on GitHub<p>

## 3. Commit py file to Git Repo
### Set Up:

- Connect Git credentials

> ```git config --global user.name "MY USERNAME"```

> ```git config --global user.email "MY EMAIL"```

- Create and add SSH key

> ```ssh-keygen -t ed25519 -C "MY EMAIL"```

> ```eval "$(ssh-agent -s)"```

> ```ssh-add ~/.ssh/id_ed25519```

> ```cat ~/.ssh/id_ed25519.pub```

- Added key to GitHub and verified connection

> ```ssh -T git@github.com```

- Clone reporisory

> ```cd ~/projects```

> ```git clone git@github.com:gina-gerlach/dsta-2025-1```

> ```cd dsta-2025-1```

> ```git checkout -b gina_milestone1```
<br>

### Clone & commit python file:

- Cloned to GitHub using curl (curl chosen because only needed the one document, not whole repository. Otherwise would have used git clone.)<p>

> ```curl -o mnist_convnet.py https://github.com/keras-team/keras-io/blob/master/examples/vision/mnist_convnet.py```

- Added file with code to Git repository with code:

> ```git add mnist_convnet.py```

> ```git commit -m "Add initial MNIST code base and updated report" -m ""Action corresponds to milestone 1 task #3. Updates to report.md reflect steps taken to complete task #3."```
<br>

### Note:

At this point I noticed my commit messages were not as clean or detailed as I would like. I modified my commits with the following codes:

> ```git log --oneline```
> ```git commit --amend -m "Title" -m "Subtitle"```
> ```git rebase -i <commit-hash from log>^```
> ```git rebase --continue```
> ```git push force```

## 4. Run code

### Set up:

- Update/upgrade packages

> ```sudo apt update && sudo apt upgrade -y```

- Installed Python

> ```sudo apt install -y python3 python3-pip git python3-venv```

- Made directory and created virtual environment for reproducibility

> ```mkdir ~/projects```

> ```cd ~/projects```

> ```python3 -m venv venv```

> ```source venv/bin/activate```

- Installed necessary packages

> ```pip install --upgrade pip```

> ```pip install tensorflow keras numpy```

- Load and run  the code

> ```python mnist_convnet.py```

### Current versions

- Python 3.10.12

- numpy 2.2.6

- keras 3.11.3

- tensorflow 2.20.0

I stored this information in "requirements.txt"

> ```pip freeze > requirements.txt```

> ```git add requirements.txt```

> ```git commit -m "Add requirements.txt for reproducibility" -m "File contains versions of packages needed to run code" ```

> ```git push origin gina_milestone1```
<br>

### Are the versions dependent on the system the code is being run on?

I tested this by running the code on my macOS terminal:

> ```mkdir ~/test_env && cd ~/test_env```

> ```git clone git@github.com:gina-gerlach/dsta-2025-1```

> ```cd dsta-2025-1```


It failed because my mac didn't have access rights, only my Ubuntu VM did, so I generated a new SSH and key for my Mac by the same process I did in my Ubuntu VM

This corrected it and the code ran successfully.

Next I set up a clean environment:

> ```python3 -m venv venv```

> ```source venv/bin/activate```

Then I installed the packages on my mac:

> ```pip install -r requirements.txt```

And ran the code:

> ```python mnist_convnet.py```

It ran without error showing that it is not dependent on the system that the code is being run on. 

## 5. Explain the code

### What is the input to and the output from the neural network

Input: input_shape = (28, 28, 1)<br>
28 pixel by 28 pixel grayscale (1 color channel) images

Output: num_classes = 10 <br>
10 classes that the image can be sorted into (0-9)

### What is Keras? And how does it relate to Tensorflow?

Keras is an API or Application Programming Interface used for building neural networks. <br>
It is high level, meaning it automates otherwise complex problems associated with building a neural network. <br>
For example, Keras has functions and classes to create layers, build models, preprocess data, and train and evaluate models.


TensorFlow on the otherhand is a low-level engine that actually perfoms the operations called via Keras. <br>
In relation to Keras, TensorFlow is backend engine that performs the computations while Keras is the user friendly interface.

### How is the data loaded

> ```(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()```

The data is loaded in with the keras dataset loader and is split into training and testing sets.

### Which dependencies are imported, what do they do?

numpy (numerical python) for array manipulation and preprocessing

keras for model building, training and evaluation

keras.layers for Convolutional Neural Network (CNN) layers

### What kind of neural network architecture are you dealing with?

A Convolutional Neural Network (CNN) or "convnet" as seen in the code when it uses the function:

> ```layers.Conv2D```

### Why do you need a neural network for this task in the first place?

The problem is a pattern recognition problem so you need a model that can learn spatial features from images.
A simpler machine learning model wouldn't work because you need someone to manually engine the features which is too complex. A neural network learns the features independently.

## 6. Add documentation file

README.md updated and pushed.

