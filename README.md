# deep-learning-kth

This repository contains the code for the course "Deep Learning in Data Sciente" at KTH Royal Institute of Technology.

## How to run the code

### Setup

First, clone the repository:

```bash
git clone https://github.com/MartinEBravo/deep-learning-kth.git
cd deep-learning-kth
```

Then, create a virtual environment and install the required packages:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Finally we generate the data for the course:

```bash
make datasets
```