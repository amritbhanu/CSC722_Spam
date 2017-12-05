## CSC722_Spam
- Github Repository can be found here: https://github.com/amritbhanu/EDM591_Hyperparameter

## Installation:
- Run 'pip install -r requirements.txt' to get the required python packages.
- Need python2.7

## Directory Structure:
- data folder contains raw data as well as preprocessed data. You will need raw spam.csv and processed_spam.csv would be generated.
- dump folder contains the results dump from running our scripts. So that results can be generated quickly.
- results folder contain all csvs. They will automatically be generated by running our scripts.
- src directory contains all our scripts.

## Src scripts:
- main.py is the main code which runs our smote and without smote results and generates dump.
- ML.py is the generalised code of all our Machine learning implementations.
- Preprocess.py is the preprocess script for spam dataset.
- read_pickle.py generates csvs/results from dumps automatically and stores in results folder.

## How to run scripts:
Go into src folder and run sequentially in the order how we mentioned below. This will take some time as we running many evaluations, with multiple features, multiple learners and with smote and without smote. It may take upto 6-8 hours to terminate.
1) 'python main.py'
2) 'python read_pickle.py'

