from torch import load, tensor
from model_architectural import BERT_Arch
import torch
import re
import joblib
import numpy as np
from transformers import BertTokenizerFast
from flask import Flask, request, render_template
from transformers import AutoModel
import sklearn


max_seq_len = 11
tokenizer = joblib.load('tokenizer')
le = joblib.load('le')
#model = load('./bert_model.pt')
#print(model.state_dict())
bert = AutoModel.from_pretrained('bert-base-uncased')
model = BERT_Arch(bert)
model.load_state_dict(torch.load('bert_model.pt')) # it takes the loaded dictionary, not the path file itself


app = Flask(__name__)






@app.route("/")
def home():
    return (render_template("index.html"))


@app.route("/get", methods=["POST", "GET"])
def get_bot_response():

    str1 = str(request.args.get('msg'))
    str1 = re.sub(r'[^a-zA-Z ]+', '', str1)
    test_text = [str1]
    model.eval()
    tokens_test_data = tokenizer(
        test_text,
        max_length=max_seq_len,
        pad_to_max_length=True,
        truncation=True,
        return_token_type_ids=False
    )
    test_seq = tensor(tokens_test_data['input_ids'])
    test_mask = tensor(tokens_test_data['attention_mask'])

    preds = None

    preds = model(test_seq, test_mask)
    preds = preds.detach().cpu().numpy()
    preds = np.argmax(preds, axis=1)
    # print('Response: ', le.inverse_transform(preds)[0])
    return str((le.inverse_transform(preds)[0]))

if __name__ == "__main__":
    app.run(debug=True)


