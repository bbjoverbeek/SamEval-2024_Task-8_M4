# SamEval-2024_Task-8_M4

## Requirements
- Shared task data for training the model
- Python 3.10 or higher
- For all the required packages, see the `requirements.txt` file
- For spacy you need to download the English language model with `python -m spacy download en_core_web_sm`

## Add data to the project
To run the code the data needs to be added to the data folder. Download the datasets from
the [Google Drive folder](https://drive.google.com/drive/folders/14DulzxuH5TDhXtviRVXsH5e2JTY2POLi). Extract the files
and make sure the folder structure looks like this:

```
./data
├── .gitkeep
├── SubtaskA
│   ├── subtaskA_dev_monolingual.jsonl
│   ├── subtaskA_dev_multilingual.jsonl
│   ├── subtaskA_train_monolingual.jsonl
│   └── subtaskA_train_multilingual.jsonl
├── SubtaskB
│   ├── subtaskB_dev.jsonl
│   └── subtaskB_train.jsonl
└── SubtaskC
    ├── subtaskC_dev.jsonl
    └── subtaskC_train.jsonl
```

This way, when code does use hardcoded filepaths, no errors will occur. If code takes command line arguments the
examples will use this structure.

## Main workflow
The main workflow is as follows:
- Create features for the training and test data with the `create_features.py` file
- Vectorize the features with the `vectorize.py` file
- Train the model and test the model with the `model.py` file

To test which features work best you can use the `run.py` file, which will run the file on all combinations of features, different traditional classifiers, and different neural networks with combinations of hyperparameters. 

The `test.py` file can be used to train the model and retrieve the predictions for the test data.

## Create features
In the examples beneath here, the features are created for all the feature options we have.

Train features:

```
python create_features.py --input data/SubtaskA/subtaskA_train_monolingual.jsonl --output features/SubtaskA/train_monolingual --features domain tense voice sentiment named-entities pronouns pos-tags dep-tags
```

Dev features:

```
python create_features.py --input data/SubtaskA/subtaskA_train_monolingual.jsonl --output features/SubtaskA/train_monolingual --features domain tense voice sentiment named-entities pronouns pos-tags dep-tags
```

The train and dev features for Subtask A are already created. They can be found
on [google drive](https://drive.google.com/drive/folders/1Xzuq8QXmhnyHFHn96p61FO3Pi7JRilpN?usp=sharing). You can also find the vectors we used for training and testing the model on the shared task data. 

For more help on the arguments, run `python create_features.py --help`

## Vectorize features
In the examples for vectorizing, only the sentiment feature is used for vectorizing.

Train features:

```
python vectorize.py --output vectors/SubtaskA/train_monolingual --input features/SubtaskA/train_monolingual --features sentiment  
```

Dev features:

```
python vectorize.py --output vectors/SubtaskA/dev_monolingual --input features/SubtaskA/dev_monolingual --features sentiment --vectorizer vectors/SubtaskA/train_monolingual     
```

For more help on the arguments, run `python vectorize.py --help`

## Training files directory structure
While in the `create_features.py` file and the `vectorize.py` file the input and output directories can be determined, not all training files are this flexible. They require the structure of the data to be as follows. 

For the features it will be:
```
features/SubtaskA/dev_monolingual/dep-tags.json
```

Where `features` is the output directory, `SubtaskA` is the subtask, `dev_monolingual` is the dataset and `dep-tags` is the feature.

For the vectors it will be:
```
vectors/SubtaskA/dev_monolingual/dep-tags/vectors.npy
```

Where `vectors` is the output directory, `SubtaskA` is the subtask, `dev_monolingual` is the dataset, `dep-tags` is the feature and `vectors.npy` is the vectorized feature.

This structure is needed for the `run.py` and `test.py` files. For the `model.py` file you can specify the input directory for both the training and test vectors. Also for the training and test data, but the internal structure of the data directory should be the same as provided by the shared task organizers.

