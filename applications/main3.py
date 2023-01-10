from nltk.corpus import webtext

if __name__ == '__main__':

    for fileid in webtext.fileids():
        print(fileid, webtext.raw(fileid)[:200], '...')