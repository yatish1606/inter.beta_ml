# from main.grammar.gr_corr import GrammarCorrection


# gr = GrammarCorrection('./main/socket/vocab.train')
# gr.infer('i has a good boy')

from main.confidence.audio_to_text import convert

file = open('Fanfare60.wav','rb')
print(convert(file.read(),160000))