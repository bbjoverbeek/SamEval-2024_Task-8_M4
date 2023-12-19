import json
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
import spacy
from tqdm.auto import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.metrics import classification_report


def main():
    # Uncomment to run sentiment analysis

    tqdm.pandas()
    # df = pd.read_json('data/tokenized/SubtaskA/subtaskA_train_monolingual.jsonl', lines=True)

    sia = SentimentIntensityAnalyzer()
    # df['sentiment'] = df['text'].progress_apply(lambda text: sia.polarity_scores(text)['compound'])
    # print(df['sentiment'])
    # pickle.dump(df, open("sentiment_tokenized.p", "wb"))

    # Load sentiment analysis data (Comment to run sentiment code)
    df = pickle.load(open("sentiment_tokenized.p", "rb"))

    # plt.figure(figsize=(8, 6))
    # sns.boxplot(x='model', y='sentiment', data=df, showfliers=False)  # ci=None removes error bars
    # plt.xlabel('Category')
    # plt.ylabel('Average Sentiment')
    # plt.title('Average Sentiment by Category')
    # plt.show()
    
    # plt.figure(figsize=(8, 6))
    # sns.boxplot(x='source', y='sentiment', data=df, showfliers=False)  # ci=None removes error bars
    # plt.xlabel('Category')
    # plt.ylabel('Average Sentiment')
    # plt.title('Average Sentiment by Source')
    # plt.show()

    model = svm.SVC(kernel='linear')
    X = df.get(['sentiment'])
    y = df['label']
    

    pipeline = Pipeline([
        ("features", FunctionTransformer(lambda items: [[x["sentiment"] ]for x in items], validate=False)),
        ("classifier", model)
    ])

    pipeline.fit(df.to_dict(orient="records"), y)

    # test_data = pd.read_json('./data/tokenized/SubtaskA/subtaskA_dev_monolingual.jsonl', lines=True)
    # test_data['sentiment'] = test_data['text'].progress_apply(lambda text: sia.polarity_scores(text)['compound'])
    # pickle.dump(test_data, open("dev_sentiment_tokenized.p", "wb"))

    test_data = pickle.load(open("dev_sentiment_tokenized.p", "rb"))

    X = test_data.get(['sentiment'])

    y_pred = pipeline.predict(test_data.to_dict(orient="records"))
    y_true = test_data['label']
    print(y_true)

    # print(y_pred)
    # print(y_true)
    print(set(y_true) - set(y_pred))

    result = classification_report(y_true, y_pred)
    print("Classification Report:\n", result)


if __name__ == "__main__":
    main()
# %%
