from flask import Flask, jsonify, request
from transformers import AutoModelForSequenceClassification, pipeline


app = Flask(__name__)

classifier_conf = pipeline(model="diegorossi/analysis-confusion-MOOC")

@app.route('/',methods=['POST'])
def inferir_sentimento():
    content = request.get_json()
    confusion = classifier_conf(content["message"])
    return jsonify(confusion[0])

if __name__ == '__main__':
    app.run(debug=True)