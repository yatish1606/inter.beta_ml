conf_model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
st_model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
st_processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")
conf_processor = Wav2Vec2Processor.from_pretrained(conf_model_name)
conf_model = EmotionModel.from_pretrained(conf_model_name)