# from main.grammar.gr_corr import GrammarCorrection


# gr = GrammarCorrection('./main/socket/vocab.train')
# gr.infer('i has a good boy')

from main.confidence.audio import infer

file = open('Fanfare.wav','r')
print(infer(file))