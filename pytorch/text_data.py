import transformers, torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup


class TextDataset(Dataset):
    
    def __init__(self, texts, labels, files, tokenizer, max_len) -> None:
        super().__init__()
        self.texts = texts
        self.labels = labels
        self.files = files
        self.tokenizer = tokenizer
        self.max_len = max_len


    def __len__(self):
        return len(self.texts)
    

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        file = self.files[item]

        encoding = self.tokenizer.encode_plus(
            text, 
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt', 
            #'tf': Return TensorFlow tf.constant objects.
            #'pt': Return PyTorch torch.Tensor objects.
            #'np': Return Numpy np.ndarray objects.
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long),
            'file': file
        }
    
    def create_data_loader(df, tokenizer, max_len, batch_size):
        ds = TextDataset(
            texts = df.text.to_numpy(),
            labels=df.label.to_numpy(),
            files = df.file.to_numpy(),
            tokenizer=tokenizer,
            max_len=max_len
        )
        return DataLoader(
            ds, 
            batch_size=batch_size,
            num_workers=4
        )

