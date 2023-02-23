import torch

def convert(audio,sampling_rate):
    speech, _ = sf.read(r"C:\Users\rohan_naik\Desktop\semicolons23_feed.back_backend\server\Fanfare60.wav")
    inputs = processor(speech, sampling_rate=16_000, return_tensors="pt")
    generated_ids = model.generate(inputs=inputs["input_values"], attention_mask=inputs["attention_mask"])
    transcription = processor.batch_decode(generated_ids)
    return transcription