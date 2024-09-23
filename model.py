import torch
from transformers import BertTokenizer, BertForSequenceClassification

def load_model(model_path):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    adjusted_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('fc.'):
            new_key = key.replace('fc.', 'classifier.')
            adjusted_state_dict[new_key] = value
        else:
            adjusted_state_dict[key] = value
    model.load_state_dict(adjusted_state_dict)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer

def predict(text, model, tokenizer):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    label = torch.argmax(probabilities, dim=1).item()
    return "Contains hate speech" if label == 1 else "Does not contain hate speech"
