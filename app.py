from flask import Flask, render_template, request, jsonify
import logging
import requests

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Hugging Face API endpoint and API key
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
headers = {"Authorization": "Bearer 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"}


def query_huggingface_api(data, min_length, max_length):
    payload = {
        "inputs": data,
        "parameters": {"min_length": min_length, "max_length": max_length},
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


def summarizer_with_huggingface_api(raw_text, summary_length_percentage):
    input_length = len(raw_text.split())
    target_summary_length = int(input_length * summary_length_percentage / 100)
    min_length = int(target_summary_length * 0.7)
    max_length = target_summary_length

    logging.info(
        f"Sending request to Hugging Face API with min_length={min_length} and max_length={max_length}")
    response = query_huggingface_api(raw_text, min_length, max_length)

    summary = response[0]['summary_text'] if response and 'summary_text' in response[0] else "Summary could not be generated"
    logging.info(f"Generated summary: {summary}")
    return summary


def get_summary_length_percentage(option):
    if option == 'detailed':
        return 50
    elif option == 'medium':
        return 25
    elif option == 'brief':
        return 10
    else:
        return 25


@app.route("/", methods=["GET"])
def index():
    return render_template('index.html')


@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.json
    raw_text = data['raw_text']
    summary_option = data['summary_option']
    summary_length_percentage = get_summary_length_percentage(summary_option)
    logging.info(f"Received text: {raw_text}")
    logging.info(f"Summary option: {summary_option}")

    try:
        summary = summarizer_with_huggingface_api(
            raw_text, summary_length_percentage)
        logging.info(f"Summary: {summary}")
        return jsonify({'summary': summary, 'word_count': len(summary.split())})
    except Exception as e:
        logging.error(f"Error generating summary: {e}")
        return jsonify({'error': 'Error generating summary'}), 500


if __name__ == '__main__':
    app.run(debug=True)
