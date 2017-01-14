txt_file = open("deneme.txt", "w")
a=1
b=2
c=3
txt_file.write("{0},{1},{2}".format(int(a), int(b), c))
txt_file.close()