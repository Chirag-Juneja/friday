"""Chatbot"""
import time
import re
import numpy as np
import tensorflow as tf

########## DATA PREPROCESSING ##########

lines = (
    open("data/movie_lines.txt", encoding="utf-8", errors="ignore").read().split("\n")
)

conversations = (
    open("data/movie_conversations.txt", encoding="utf-8", errors="ignore")
    .read()
    .split("\n")
)

id2line = {}
for line in lines:
    _line = line.split(" +++$+++ ")
    if len(_line) == 5:
        id2line[_line[0]] = _line[-1]

conversations_ids = []
for conversation in conversations[:-1]:
    _conversation = (
        conversation.split(" +++$+++ ")[-1][1:-1].replace("'", "").replace(" ", "")
    )
    conversations_ids.append(_conversation.split(","))

questions = []
answers = []
for converstion in conversations_ids:
    for i in range(len(converstion) - 1):
        questions.append(id2line[converstion[i]])
        answers.append(id2line[converstion[i + 1]])


def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
    return text


clean_questions = [clean_text(question) for question in questions]

clean_answers = [clean_text(answer) for answer in answers]

word2count = {}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

threshold = 20
questionswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        questionswords2int[word] = word_number
        word_number += 1

answerswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        answerswords2int[word] = word_number
        word_number += 1

tokens = ["<PAD>", "<EOS>", "<OUT>", "<SOS>"]
for token in tokens:
    questionswords2int[token] = len(questionswords2int) + 1
    answerswords2int[token] = len(answerswords2int) + 1

answersints2word = {w_i: w for w, w_i in answerswords2int.items()}

clean_answers = [ans + " <EOS>" for ans in clean_answers]

questions_into_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questionswords2int:
            ints.append(questionswords2int["<OUT>"])
        else:
            ints.append(questionswords2int[word])
    questions_into_int.append(ints)

answers_into_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answerswords2int:
            ints.append(answerswords2int["<OUT>"])
        else:
            ints.append(answerswords2int[word])
    answers_into_int.append(ints)

sorted_clean_questions = []
sorted_clean_answers = []

for length in range(1, 25 + 1):
    for index, question in enumerate(questions_into_int):
        if len(question) == length:
            sorted_clean_questions.append(questions_into_int[index])
            sorted_clean_answers.append(answers_into_int[index])

########## BUILDING SEQ2SEQ MODEL ##########