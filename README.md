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