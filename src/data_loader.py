import spacy
import re
import math

import pandas as pd

nlp = spacy.load("en_core_web_lg")
SEED = 42

def tokenize(text):
    """
    Processes the input text, splits it into sentences, and further processes each sentence
    to extract non-numeric words. It constructs a list of these words for each sentence.

    Parameters:
    text (str): A string containing multiple sentences.

    Returns:
    list: A list of lists, where each inner list contains the words from one sentence,
          excluding any numeric strings.
    """
    # remove newline characters, this line is not necessary for all cases
    # the reason it is included here is because the abstracts in the dataset contain abnormal newline characters
    # e.g. Recent works on diffusion models have demonstrated a strong capability for\nconditioning image generation,
    text=text.replace('\n',' ')
    # Initialize an empty list to store the list of words for each sentence
    sentence_list=[]
    # Process the sentence using the spacy model to extract linguistic features and split into components
    doc=nlp(text)
    # Iterate over each sentence in the processed text
    for sent in doc.sents:
        # Extract the words from the sentence
        words = re.findall(r'\b\w+\b', sent.text.lower())
        # Remove any words that are numeric
        words_without_digits=[word for word in words if not word.isdigit()]
        # If the list is not empty, append the list of words to the sentence_list
        if len(words_without_digits)!=0:
            sentence_list.append(words_without_digits)
    return sentence_list

def tokenize_df(series):
    all_sentences = [tokenize(text) for text in series]

def create_data_split(data, val_ratio, text_sources=["Abstract", "Introduction", "Related Work", "Methods", "Result&Discussion", "Conclusion"], include_pos="all"):
    train_texts = []
    val_texts = []
    data["is_val"] = False
    data.loc[data.sample(math.floor(data.length * val_ratio), random_state=SEED).index, "is_val"] = True
    for row in data.iterrows():
        paper_texts = []
        for source in text_sources:
            paper_texts+=tokenize(row[source])
        if row["is_val"]:
            val_texts+=paper_texts
        else:
            train_texts+=paper_texts
    return train_texts, val_texts

                

def create_splits():
    pass

def get_source_data():
    retraction_df = pd.read_parquet("../retraction_fulltext_dataset/24_08_22_retraction_with_text.gzip")
    reference_df = pd.read_parquet("../retraction_fulltext_dataset/24_11_30_reference_articles.gzip")
    return retraction_df, reference_df

def dataset_gpt_inference(df):
    df["OriginalPaperDate"] = pd.to_datetime(df["OriginalPaperDate"], format='%m/%d/%Y %H:%M')
    target_date = pd.to_datetime('29.11.2022', format='%d.%m.%Y')
    df = df[df["OriginalPaperDate"] > target_date]
    paper_texts = []
    for paper in df.itertuples():
        if paper.Abstract is not None:
            paper_text = " ".join(paper.Abstract)
            paper_texts+=tokenize(paper_text)
    return paper_texts

if __name__=="__main__":
    retraction_df, reference_df = get_source_data()
    retraction_gpr_inference_dataset = dataset_gpt_inference(retraction_df)
