from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, TrainingArguments, Trainer
import torch

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained('gpt2')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

dados_treinamento = list()
with open("dataset", "r", encoding='utf-8') as arquivo:
    for linha in arquivo:
        linha = linha.strip()
        _input, _output = linha.split(':', 1)
        dados_treinamento.append((_input, _output))

class MeuDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, dados_treinamento):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []

        for entrada, saida in dados_treinamento:
            encodings_dict = tokenizer('<|startoftext|>'+ entrada + saida + '<|endoftext|>', truncation=True, max_length=512, padding="max_length")
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
            self.labels.append(torch.tensor(encodings_dict['input_ids']))
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {'input_ids': self.input_ids[idx], 'attention_mask': self.attn_masks[idx], 'labels': self.labels[idx]}

dataset = MeuDataset(tokenizer, dados_treinamento)

training_args = TrainingArguments(
    output_dir='./results',         
    num_train_epochs=10,             
    per_device_train_batch_size=4,   
    per_device_eval_batch_size=4,   
    warmup_steps=500,                
    weight_decay=0.01,               
    logging_dir='./logs',            
    learning_rate=0.005
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

model.save_pretrained('./meu_modelo_treinado')
tokenizer.save_pretrained('./meu_modelo_treinado')
