import os

path = './tex/'
files = os.listdir(path)
count = 0
deffile = []
for i in range(len(files)):
    file = files[i].split('.')
    index = 0
    if len(file) > 1:
        if file[1] == 'tex':
            #print(files[i])
            f = open(path+files[i], "r")
            rows = f.read().split('\n')
            #print(rows)
            for j in range(len(rows)):
                if count != 0:
                    if rows[j] == "\\tableofcontents":
                        #print(rows[j])
                        j += 1
                #j = index
                #print(rows)
                #print(rows[j])
                ##controllare ultima parte
                if count < (len(files)-1):
                    if rows[j] != '\\input{chapters}':
                        #print(rows[j])
                        deffile.append(rows[j])
                    else:
                        deffile.append(rows[j])
            f.close()
        count += 1
f = open("def.tex", "w")
for i in range(len(deffile)):
    f.write(deffile[i])
    f.write('\n')
f.close()
#print(deffile)

