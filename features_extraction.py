import cv2
import numpy as np
import math
from scipy.spatial import distance
from math import pi
from sklearn.feature_extraction.text import TfidfVectorizer
import emoji
import re
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd

max_features = 20
vectorizer = TfidfVectorizer(max_features=max_features)


def remove_emojies(input):
    cleaned_string = emoji.demojize(input)
    text_only = re.sub(r":[a-zA-Z_]+:", "", cleaned_string)
    return text_only


def format_paragraph(text):
    text = text.replace("..", ". ")
    text = " ".join(text.split())
    sentences = text.split(". ")
    paragraph = ". ".join(sentences)
    return paragraph


def convert_para(descriptions):
    para = ""
    for description in descriptions:
        description = remove_emojies(description)
        para += f"{description}."
    para = format_paragraph(para)
    return para


def summarize_para(paragraph):
    # Load pre-trained GPT-2 model and tokenizer
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # Tokenize and encode the paragraph
    input_ids = tokenizer.encode(paragraph, return_tensors="pt")
    # Generate the summary
    output = model.generate(input_ids, max_length=200, num_return_sequences=1)
    # Decode the generated summary
    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    return summary


def get_features_description(description):
    documents = [description]
    tfidf_matrix = vectorizer.fit_transform(documents)
    return tfidf_matrix


def set_description_features(dataframe, tfidf_matrix):
    for i in range(max_features):
        value = tfidf_matrix[0, i] if i < tfidf_matrix.shape[1] else 0
        dataframe[f"description_feature_{i}"].append(value)
    return dataframe


def extract_features(description, house_size, bedrooms, bathrooms, land_size, type):
    # Initialize dataframe with empty lists
    dataframe = {
        "House size:": [house_size],
        "Bedrooms:": [bedrooms],
        "Bathrooms:": [bathrooms],
        "Land size:": [land_size],
    }
    # Initialize description features with empty lists
    for i in range(max_features):
        dataframe[f"description_feature_{i}"] = []
    # Get and set description features
    tfidf_matrix = get_features_description(description)
    dataframe = set_description_features(dataframe, tfidf_matrix)
    dataframe["type"] = [type]
    # Convert to DataFrame
    dataframe = pd.DataFrame(dataframe)
    return dataframe
