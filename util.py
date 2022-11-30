def writetofile(f, text):
    with open(f, 'a') as f:
        f.write(text)
        f.write('\n')