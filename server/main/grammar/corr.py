from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("addy88/t5-grammar-correction")

model = AutoModelForSeq2SeqLM.from_pretrained("addy88/t5-grammar-correction")

def infer(sentence):
    input_ids = tokenizer(f'grammar: {sentence}.', return_tensors='pt').input_ids
    outputs = model.generate(input_ids)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
