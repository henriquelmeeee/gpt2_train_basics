from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model = GPT2LMHeadModel.from_pretrained('./meu_modelo_treinado')
tokenizer = GPT2Tokenizer.from_pretrained('./meu_modelo_treinado')

while True:
    input_ids = tokenizer.encode(str(input("Texto: ")), return_tensors='pt')

    outputs = model.generate(input_ids, max_length=100)

    texto_gerado = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(texto_gerado)
