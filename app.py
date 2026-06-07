import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import re
import time

# Reduce TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# TensorFlow optimization
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")


# ================= CONFIG =================
class ModelConfig:
    MAX_SEQ_LEN_INPUT = 20
    MAX_SEQ_LEN_OUTPUT = 20
    HIDDEN_UNITS = 256


# ================= LOAD MODELS =================
@st.cache_resource
def load_models():
    encoder_model = load_model(
        os.path.join(MODELS_DIR, "encoder_model.h5")
    )

    decoder_model = load_model(
        os.path.join(MODELS_DIR, "decoder_model.h5")
    )

    with open(
        os.path.join(MODELS_DIR, "input_tokenizer.pickle"),
        "rb"
    ) as f:
        input_tokenizer = pickle.load(f)

    with open(
        os.path.join(MODELS_DIR, "output_tokenizer.pickle"),
        "rb"
    ) as f:
        output_tokenizer = pickle.load(f)

    config = ModelConfig()

    return (
        encoder_model,
        decoder_model,
        input_tokenizer,
        output_tokenizer,
        config,
    )


# ================= CLEAN TEXT =================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s\u0900-\u097F]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ================= TRANSLATE =================
def translate(
    text,
    encoder_model,
    decoder_model,
    input_tokenizer,
    output_tokenizer,
    config,
):
    cleaned = clean_text(text)

    sequence = input_tokenizer.texts_to_sequences([cleaned])

    padded = pad_sequences(
        sequence,
        maxlen=config.MAX_SEQ_LEN_INPUT,
        padding="post",
    )

    states_value = encoder_model.predict(
        padded,
        verbose=0
    )

    if isinstance(states_value, tuple):
        states_value = list(states_value)

    if not isinstance(states_value, list):
        states_value = [states_value]

    start_token = output_tokenizer.word_index.get("start")

    if start_token is None:
        start_token = list(
            output_tokenizer.word_index.values()
        )[0]

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = start_token

    decoded_sentence = []

    for _ in range(config.MAX_SEQ_LEN_OUTPUT):

        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value,
            verbose=0,
        )

        sampled_token_index = int(
            np.argmax(output_tokens[0, -1, :])
        )

        sampled_word = output_tokenizer.index_word.get(
            sampled_token_index,
            "",
        )

        if sampled_word in ["end", ""]:
            break

        decoded_sentence.append(sampled_word)

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        states_value = [h, c]

    return " ".join(decoded_sentence)


# ================= STREAMLIT UI =================
st.set_page_config(
    page_title="English → Hindi Translator",
    page_icon="🌐",
    layout="centered",
)

st.title("🌐 English → Hindi Translator")
st.write("Translate English text into Hindi")

with st.spinner("Loading models..."):
    (
        encoder_model,
        decoder_model,
        input_tokenizer,
        output_tokenizer,
        config,
    ) = load_models()

text = st.text_area(
    "Enter English Text",
    height=150,
    placeholder="Example: What are you doing?"
)

if st.button("Translate"):

    if text.strip():

        with st.spinner("Translating..."):

            start = time.time()

            result = translate(
                text,
                encoder_model,
                decoder_model,
                input_tokenizer,
                output_tokenizer,
                config,
            )

            st.success(result)

            st.caption(
                f"Completed in {time.time() - start:.2f} seconds"
            )

    else:
        st.warning("Please enter some text.")