from flask import Flask, render_template, request, jsonify, session, send_file
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from datetime import datetime
import pandas as pd

app = Flask(__name__)
app.secret_key = 'super-secret-key'

class ModelConfig:
    MAX_SEQ_LEN_INPUT = 20
    MAX_SEQ_LEN_OUTPUT = 20
    HIDDEN_UNITS = 256

# ================= LOAD MODELS =================
def load_models():
    encoder_model = load_model('saved_models/encoder_model.h5')
    decoder_model = load_model('saved_models/decoder_model.h5')

    with open('saved_models/input_tokenizer.pickle', 'rb') as f:
        input_tokenizer = pickle.load(f)

    with open('saved_models/output_tokenizer.pickle', 'rb') as f:
        output_tokenizer = pickle.load(f)

    with open('saved_models/config.pickle', 'rb') as f:
        config = pickle.load(f)

    print("✅ Models loaded successfully!")
    return encoder_model, decoder_model, input_tokenizer, output_tokenizer, config


# ================= CLEAN TEXT =================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s\u0900-\u097F]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ================= TRANSLATE FUNCTION =================
def translate(text, encoder_model, decoder_model, input_tokenizer, output_tokenizer, config):
    cleaned = clean_text(text)

    sequence = input_tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=config.MAX_SEQ_LEN_INPUT, padding='post')

    states_value = encoder_model.predict(padded, verbose=0)

    # FIX: ensure list
    if isinstance(states_value, tuple):
        states_value = list(states_value)
    if not isinstance(states_value, list):
        states_value = [states_value]

    # FIX: start token
    start_token = output_tokenizer.word_index.get('start')
    if start_token is None:
        start_token = list(output_tokenizer.word_index.values())[0]

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = start_token

    decoded_sentence = []

    for _ in range(config.MAX_SEQ_LEN_OUTPUT):
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value, verbose=0
        )

        sampled_token_index = int(np.argmax(output_tokens[0, -1, :]))
        sampled_word = output_tokenizer.index_word.get(sampled_token_index, '')

        if sampled_word == 'end' or sampled_word == '':
            break

        decoded_sentence.append(sampled_word)

        # RESET target
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        states_value = [h, c]

    return ' '.join(decoded_sentence)


# ================= INIT =================
print("🚀 Loading models...")
encoder_model, decoder_model, input_tokenizer, output_tokenizer, config = load_models()


# ================= ROUTES =================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/translate', methods=['POST'])
def translate_api():
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'Invalid JSON'}), 400

        english_text = data.get('text', '').strip()

        if not english_text:
            return jsonify({'error': 'No text provided'}), 400

        hindi_text = translate(
            english_text,
            encoder_model,
            decoder_model,
            input_tokenizer,
            output_tokenizer,
            config
        )

        return jsonify({
            'english': english_text,
            'hindi': hindi_text,
            'success': True
        })

    except Exception as e:
        print("❌ TRANSLATE ERROR:", e)
        return jsonify({'error': str(e)}), 500


@app.route('/history')
def get_history():
    return jsonify(session.get('history', []))


@app.route('/clear_history', methods=['POST'])
def clear_history():
    session['history'] = []
    return jsonify({'success': True})


@app.route('/export_history')
def export_history():
    history = session.get('history', [])

    if not history:
        return jsonify({'error': 'No history'}), 400

    df = pd.DataFrame(history)
    file_path = 'translations.csv'
    df.to_csv(file_path, index=False)

    return send_file(file_path, as_attachment=True)


# ================= RUN =================
if __name__ == '__main__':
    app.run(debug=True, port=5000)