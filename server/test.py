from main.grammar.corr import infer


print(infer('i has a good boy'))

# from main.confidence.audio import infer

# file = open('Fanfare.wav','r')
# print(infer(file)
# def main():
#   #open file in read mode
#   f = open("MeetinginGeneral.vtt")


#   #store as string
#   transcript_string = str(f.read())
#   #print("Transcript: \n",transcript_string)

#   #store the transcript as list
#   string_list2 = list()
#   string_list2 = transcript_string.split("\n")
#   res_list = list()


#   for i in range(0,len(string_list2)):
#     if '-' not in string_list2[i]:
#       if string_list2[i] != '':
#         res_list.append(string_list2[i])

#   print("Transcript \n",res_list)

#   f.close()

# if __name__ == "__main__":
#   main()

