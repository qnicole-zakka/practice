from transformers import BertTokenizer, BertModel, BertConfig, AutoTokenizer, AutoModelForMaskedLM


tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModelForMaskedLM.from_pretrained("roberta-base")

input_ids, attention_mask = [], []

for s in sentences: 
    encoded_dict = tokenizer.encode_plus(
                        s,
                        add_special_tokens=True,
                        max_length=64,
                        pad_to_max_length=True,
                        return_tensors='pt',
    )
    input_ids.append(encoded_dict['input_ids'])
    attention_mask.append(encoded_dict['attention_mask'])   

 