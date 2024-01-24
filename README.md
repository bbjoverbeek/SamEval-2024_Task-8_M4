# SamEval-2024_Task-8_M4

## Add data to the project
To run the code the data needs to be added to the data folder. Download the datasets from the [Google Drive folder](https://drive.google.com/drive/folders/14DulzxuH5TDhXtviRVXsH5e2JTY2POLi). Extract the files and make sure the folder structure looks like this:

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

This way, when code does use hardcoded filepaths, no errors will occur. If code takes command line arguments the examples will use this structure. 

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

The train and dev features for Subtask A are already created. They can be found on [google drive](https://drive.google.com/drive/folders/1Xzuq8QXmhnyHFHn96p61FO3Pi7JRilpN?usp=sharing)

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