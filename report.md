# Milestone 1 - Report
<p>Datascience Toolkits and Architectures<br>
Gina Gerlach<br>
**Base set up:**
- Created an Ubuntu 22.04 (ARM64) virtual machine using UTM on macOS.
- Allocated 4 GB RAM and set up a shared folder for file transfer between macOS and Ubuntu.
- Installed Python, Git, and other necessary tools<br>
`sudo apt install -y python3 python3-pip git python3-venv wget unzip build-essential curl`<p>


## 1. Dataset Description
<p>The MNIST handwritted digit database is an image collection of handwritten digits (numbers 0-9).<br>
- Size: 18.2MB
- Number of rows: 70,000
- Type: 28x28px black and white images
- Source: Census Bureau employees and high school students (evenly distributed between testing and training sets)
- Training set: 60,000 
- Testing set: 10,000 <br>
- Type of problem: Classification
- Number of classes: 10 (1 per digit)
- Images per class: 7,000 (6,000 training, 1,000 testing) <br>
This database is used for creating pattern recognition methods with machine learning models.<p>

## 2. Check out code base
<p>Viewed on GitHub<p>

## 3. Commit py file to Git Repo
<p>**Set Up:**<br>
- Connect Git credentials
`git config --global user.name "MY USERNAME"`
`git config --global user.email "MY EMAIL"`
- Create and add SSH key
`ssh-keygen -t ed25519 -C "MY EMAIL"`
`eval "$(ssh-agent -s)"`
`ssh-add ~/.ssh/id_ed25519`
`cat ~/.ssh/id_ed25519.pub`
- Added key to GitHub and verified connection
`ssh -T git@github.com`
- Clone reporisory
`cd ~/projects`
`git clone git@github.com:gina-gerlach/dsta-2025-1`
`cd dsta-2025-1`
`git checkout -b gina_milestone1`<p>


<p>**Clone & commit python file:**<br>
- Cloned to GitHub using curl (curl chosen because only needed the one document, not whole repository. Otherwise would have used git clone.)<p>
`curl -o mnist_convnet.py https://github.com/keras-team/keras-io/blob/master/examples/vision/mnist_convnet.py`
- Added file with code to Git repository with code:<br> 
`git add mnist_convnet.py`
`git commit -m "Add initial MNIST code base and updated report" -m ""Action corresponds to milestone 1 task #3. Updates to report.md reflect steps taken to complete task #3."`<br>


<p>**Note:**<br>
At this point I noticed my commit messages were not as clean or detailed as I would like. I modified my commits with the following code:
`git log --oneline`
`git commit --amend -m "Title" -m "Subtitle"`
`git rebase -i <commit-hash from log>^`
`git rebase --continue`
`git push force`<p>

## 4. Run code
<p>**Set up:**<br>
- Update/upgrade packages <br>
`sudo apt update && sudo apt upgrade -y`
- Installed Python
`sudo apt install -y python3 python3-pip git python3-venv`
- Made directory and created virtual environment for reproducibility
`mkdir ~/projects`
`cd ~/projects`
`python3 -m venv venv`
`source venv/bin/activate`
- Installed necessary packages
`pip install --upgrade pip`
`pip install tensorflow keras numpy`
- Load and run  the code
`python mnist_convnet.py`<p>

<p>**Current versions**<br>
- Python 3.10.12
- nump 2.2.6
- keras 3.11.3
- tensorflow 2.20.0 <p>
