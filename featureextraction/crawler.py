#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import nltk
import re
import glob, os
from urlextract import URLExtract
import textstat
import emojis
import pandas as pd

class Metrics:
    textstat.set_lang("it")

    def readf(self, tweet):
        # global name
        # print(nome)
        # in_file = open(nome+".txt","r")
        # text=in_file.read()
        # in_file.close()
        # return text
        return tweet.full_text

    def addHeadingToFile(self, nome):

        nome.write("Author\t")
        # out_file.write("Testo\t")
        nome.write("NumberOfTotalCharacters\t")
        nome.write("NumberOfUppercaseCharacters\t")
        nome.write("NumberOfLowercaseCharacters\t")
        nome.write("NumberOfSpecialCharacters\t")
        nome.write("NumberOfNumbers\t")
        nome.write("NumberOfBlanks\t")
        nome.write("NumberOfWords\t")
        nome.write("LengthOfWords\t")
        nome.write("NumberOfPropositions\t")
        nome.write("PropositionsLength\t")
        nome.write("NumberOfPunctuationCharacters\t")
        nome.write("NumberOfLowercaseWords\t")
        nome.write("NumberOfUppercaseWords\t")
        nome.write("VocabularyRichness\t")
        nome.write("NumberOfURLs\t")
        nome.write("parole piu comuni\t")
        nome.write("Flesch Kincaid Grade Level\t")  # da qui alla fine sono metriche di leggibilita
        nome.write("Flesch Reading Ease formula\t")
        nome.write("Dale Chall Readability\t")
        nome.write("Automated Readability Index\t")
        nome.write("Coleman Liau Index\t")
        nome.write("Gunning Fog\t")
        nome.write("SMOG(Simple Measure of Gobbledygook)\t")
        nome.write("Linsear Write" + "\n")

    # I split the text into tokens
    def createToken(self, tweet):
        # text=self.readf(tweet)
        tokens = nltk.word_tokenize(tweet)  # divide il testo in token
        return tokens

    def createSentToken(self, tweet):
        # text=self.readf(tweet)
        tokens = nltk.sent_tokenize(tweet)  # dividei l testo in token di frasi
        return tokens

    # total number of characters excluding whitespace only
    def totalCharacter(self, tweet):
        # total=0
        # tokens=self.createToken()
    
        # for i in range(len(tokens)):
        #   total=total+len(tokens[i])
        # return total
        c = str(tweet)
        return len(c)

        # total number of uppercase characters excluding whitespace only

    def CharacterUpper(self, tweet):
        total = 0
        tokens = self.createToken(tweet)
       
        for i in range(len(tokens)):
            tok = tokens[i]
            for j in range(len(tok)):
                if tok[j].isupper():
                    total += 1
        return total

        # total number of lowercase characters excluding whitespace only

    def CharacterLower(self, tweet):
        total = 0
        tokens = self.createToken(tweet)
        for i in range(len(tokens)):
            tok = tokens[i]
            for j in range(len(tok)):
                if tok[j].islower():
                    total += 1
        return total

    # number of numbers
    def NumberCount(self, tweet):
        total = 0
        tokens = self.createToken(tweet)
        for i in range(len(tokens)):
            tok = tokens[i]
            if tok.isdecimal():
                total += 1
            else:
                for j in range(len(tok)):
                    if tok[j].isdecimal():
                        total += 1
        return total

        # number of white spaces

    def Whitespace(self, tweet):
        total = 0
        # text=self.readf(tweet)
        for i in range(len(tweet)):
            if tweet[i].isspace():
                total += 1
        return total

        # total number of special characters excluding whitespace only

    def CharacterSpecial(self, tweet):
        #  total=0
        #  regex = re.compile('[@_#$%^&*/\|~%]')
        # tokens=self.createToken()
        # for i in range(len(tokens)):
        #   tok=tokens[i]
        #  for j in range(len(tok)):
        #     if regex.search(tok[j]):
        #       total+=1
        return self.totalCharacter(tweet) - self.CharacterUpper(tweet) - self.CharacterLower(tweet) - self.NumberCount(
            tweet) - self.Whitespace(tweet) - self.NumberEmoji(tweet)

    # number of words
    def NumberWord(self, tweet):
        total = 0
        # text=self.readf(tweet)
        # parses the string starting from the first character, when it finds a space it stops and inserts the portion of the string into a list (vector).
        a = tweet.split(" ")
        # the list created could also contain words composed of an empty value. The for loop performs a check and counts only the values ​​that contain at least one character other than space.
        for i in a:
            if (i != ""):
                total += 1
        return total

    # number of emojis
    def NumberEmoji(self, tweet):
        # text=self.readf(tweet)
        return emojis.count(tweet)

    # number of propositions
    def NumberPhrase(self, tweet):
        total = 0
        tokens = self.createSentToken(tweet)
        total = len(tokens)
        return total

    # punctuation frequency
    def PunctuationFrequency(self, tweet):
        total = 0
        # text=self.readf(tweet)
        dist = nltk.FreqDist(tweet)
        total = dist['.'] + dist[','] + dist[';'] + dist[':'] + dist['!'] + dist['?'] + dist['('] + dist[')'] + dist[
            '<'] + dist['>'] + dist['-'] + dist['...'] + dist['\"'] + dist['\'']
        return total

    # most common words
    def CommonWord(self, tweet):
        nopunt = []  # list without punctuation

        lista = []
        tokens = self.createToken(tweet)
        #  print(tokens)
        stop_words = nltk.corpus.stopwords.words()
        regex = re.compile('[`.,’:!?()\-<>"\'@_#\$%\^&\*~%}{\[\]]')
        # text=self.readf(tweet)
        extractor = URLExtract()
        urls = extractor.find_urls(tweet)
        # print(nltk.word_tokenize (str(urls)))
        em = emojis.get(tweet)
        #  print(em)
        # print(stop_words)
        # for i in range(len(tokens)):
        for i in tokens:

            #  if tokens[i]!='.'and tokens[i]!=','and tokens[i]!=';'and tokens[i]!=':' and tokens[i]!='!' and tokens[i]!='?' and tokens[i]!='('  and tokens[i]!=')' and tokens[i]!='<' and tokens[i]!='>' and tokens[i]!='-' and tokens[i]!='...' and tokens[i]!='-' and tokens[i]!='\"' and tokens[i]!='\'' and not regex.match(tokens[i]) and tokens[i] not in em and tokens[i] not in urls:
            if (not regex.match(i) and not i in em and not i in str(urls)):
                # if i!='.'and i!=','and i!=';'and i!=':' and i!='!' and i!='?' and i!='('  and i!=')' and i!='<' and i!='>' and i!='-' and i!='...' and i!='"' and i!="'" and i!="“" and i!="’"  and i!="''" and not regex.match(i) and i not in em and not i in str(urls):
                nopunt.append(i)

        for i in nopunt:
            if not i.lower() in stop_words:
                lista.append(i)
        # print(lista)
        dist = nltk.FreqDist(lista)
        com = dist.most_common(5)
        return com

    # number of uppercase words
    def WordUpper(self, tweet):
        total = 0
        tokens = self.createToken(tweet)
      
        for i in tokens:
            if i.isupper():
                total += 1
        return total

    # number of lowercase words
    def WordLower(self, tweet):
        total = 0
        tokens = self.createToken(tweet)
        for i in tokens:
            if i.islower():
                total += 1
        return total

    # LengthOfWords
    def AvgLenWord(self, tweet):
        mid = 0
        somma = 0
        nopunt = []  
        lista = []  
        # text=self.readf(tweet)
        tokens = self.createToken(tweet)
        # print(tokens)
        regex = re.compile('[`.,’:!?()\-<>"\'@_#\$%\^&\*~%}{\[\]]')
        extractor = URLExtract()
        urls = extractor.find_urls(tweet)
        em = emojis.get(tweet)
        stop_words = nltk.corpus.stopwords.words()
        # print(stop_words)
        for i in tokens:
            if (not regex.match(i) and not i in em and not i in str(urls)):
                nopunt.append(i)
        for i in nopunt:
            if not i.lower() in stop_words:
                lista.append(i)
        # print(len(lista))
        for i in range(len(lista)):
            somma += len(lista[i])
        dim = len(lista)
        if dim == 0:
            dim = 1
        mid = somma / dim
        return round(mid, 2)

        # average sentence length

    def AvgLenPhrase(self, tweet):
        mid = 0
        somma = 0
        dimphr = []  
        phras = self.createSentToken(tweet)
        totalPhras = len(phras)
        for i in range(len(phras)):
            total = 0
            a = phras[i].split(" ")
            for i in a:
                if (i != ""):
                    total += 1
            dimphr.append(total)
        # print(dimphr)
        for i in dimphr:
            somma += i
        # print(somma)
        mid = somma / totalPhras
        return round(mid, 2)

        # VocabularyRichness

    def VocabularyWealth(self, tweet):
        # text=self.readf()

        nopunt = []  # lista non contenente punteggiature
        lista = []
        tokens = self.createToken(tweet)
        regex = re.compile('[`.,’:!?()\-<>"\'@_#\$%\^&\*~%}{\[\]]')
        em = emojis.get(tweet)

        stop_words = nltk.corpus.stopwords.words()
        # print(stop_words)
        for i in tokens:
            if (not regex.match(i) and not i in em):
                nopunt.append(i)
        # print(round(len(set(nopunt))/len(nopunt),2))
        for i in nopunt:
            if not i.lower() in stop_words:
                lista.append(i)
        # print(lista)
        if len(lista) == 0:
            return 0

        return round(len(set(lista)) / len(lista), 2)

    # number of url
    def UrlNumber(self, tweet):
        # text=self.readf(tweet)
        extractor = URLExtract()
        urls = extractor.find_urls(tweet)
        return len(urls)

    # Flesch Reading Ease formula
    def Fre(self, tweet):
        # text=self.readf(tweet)

        return round(textstat.flesch_reading_ease(tweet), 2)

        # Flesch Kincaid Grade Level

    def Fkgl(self, tweet):
        # text=self.readf(tweet)
        return round(textstat.flesch_kincaid_grade(tweet), 2)

        # Automated Readability Index (ARI)

    def Ari(self, tweet):
        # text=self.readf(tweet)
        return round(textstat.automated_readability_index(tweet), 2)

        # Coleman Liau Index

    def Cli(self, tweet):
        # text=self.readf(tweet)
        return round(textstat.coleman_liau_index(tweet), 2)

        # Gunning Fog

    def Gf(self, tweet):
        #  text=self.readf(tweet)
        return round(textstat.gunning_fog(tweet), 2)

        # Dale-Chall Readability Score

    def Dcr(self, tweet):
        # text=self.readf(tweet)
        return textstat.dale_chall_readability_score(tweet)

    # SMOG
    def Smog(self, tweet):
        # text=self.readf(tweet)
        return round(textstat.smog_index(tweet), 2)

    # Linsear Write
    def Lw(self, tweet):
        # text=self.readf(tweet)
        return round(textstat.linsear_write_formula(tweet), 2)

    def prova(self, twet, id):
        print("Id:" + id)
        print("NumberOfTotalCharacters:" + str(self.totalCharacter(twet).__str__()) + "\t")
        print("NumberOfUppercaseCharacters:" + str(self.CharacterUpper(twet)) + "\t")
        print("NumberOfLowercaseCharacters:" + str(self.CharacterLower(twet)) + "\t")
        print("NumberOfSpecialCharacters:" + str(self.CharacterSpecial(twet)) + "\t")
        print("NumberOfNumbers:" + str(self.NumberCount(twet)) + "\t")
        print("NumberOfBlanks:" + str(self.Whitespace(twet)) + "\t")
        print("NumberOfWords:" + str(self.NumberWord(twet)) + "\t")
        print("LengthOfWords:" + str(self.AvgLenWord(twet)) + "\t")
        print("NumberOfPropositions:" + str(self.NumberPhrase(twet)) + "\t")
        print("PropositionsLength:" + str(self.AvgLenPhrase(twet)) + "\t")
        print("NumberOfPunctuationCharacters:" + str(self.PunctuationFrequency(twet)) + "\t")
        print("NumberOfLowercaseWords:" + str(self.WordLower(twet)) + "\t")
        print("NumberOfUppercaseWords:" + str(self.WordUpper(twet)) + "\t")
        print("VocabularyRichness:" + str(self.VocabularyWealth(twet)) + "\t")
        print("NumberOfURLs:" + str(self.UrlNumber(twet)) + "\t")
        print("Flesch Kincaid Grade Level:" + str(self.Fkgl(twet)) + "\t")
        print("Flesch Reading Ease formula:" + str(self.Fre(twet)) + "\t")
        print("Dale Chall Readability:" + str(self.Dcr(twet)) + "\t")
        print("Automated Readability Index:" + str(self.Ari(twet)) + "\t")
        print("Coleman Liau Index:" + str(self.Cli(twet)) + "\t")
        print("Gunning Fog:" + str(self.Gf(twet)) + "\t")
        print("SMOG(Simple Measure of Gobbledygook):" + str(self.Smog(twet)) + "\t")
        print("Linsear Write:" + str(self.Lw(twet)) + "\n")

    def CreateTable(self, nome):
        line_count = 0
        filenumb = 1
        os.chdir("C:/Users/sonia/OneDrive/Desktop/Bot/metricheReddit/")
        for file in glob.glob("*"):
            out_file = open("C:/Users/sonia/OneDrive/Desktop/" + str(filenumb) + ".csv", 'a',
                            encoding='utf-8')
            filenumb += 1
            print("file:" + str(filenumb))
            out_file.write("Author" + ',' + '\t')
            out_file.write("NumberOfTotalCharacters" + ',' + '\t')
            out_file.write("NumberOfUppercaseCharacters" + ',' + '\t')
            out_file.write("NumberOfLowercaseCharacters" + ',' + '\t')
            out_file.write("NumberOfSpecialCharacters" + ',' + '\t')
            out_file.write("NumberOfNumbers" + ',' + '\t')
            out_file.write("NumberOfBlanks" + ',' + '\t')
            out_file.write("NumberOfWords" + ',' + '\t')
            out_file.write("LengthOfWords" + ',' + '\t')
            out_file.write("NumberOfPropositions" + ',' + '\t')
            out_file.write("PropositionsLength" + ',' + '\t')
            out_file.write("NumberOfPunctuationCharacters" + ',' + '\t')
            out_file.write("NumberOfLowercaseWords" + ',' + '\t')
            out_file.write("NumberOfUppercaseWords" + ',' + '\t')
            out_file.write("VocabularyRichness" + ',' + '\t')
            out_file.write("NumberOfURLs" + ',' + '\t')
            out_file.write("Flesch Kincaid Grade Level" + ',' + '\t')  # da qui alla fine sono metriche di leggibilita
            out_file.write("Flesch Reading Ease formula" + ',' + '\t')
            out_file.write("Dale Chall Readability" + ',' + '\t')
            out_file.write("Automated Readability Index" + ',' + '\t')
            out_file.write("Coleman Liau Index" + ',' + '\t')
            out_file.write("Gunning Fog" + ',' + '\t')
            out_file.write("SMOG(Simple Measure of Gobbledygook)" + ',' + '\t')
            out_file.write("Linsear Write" + "\n")
            line_count += 1
            autore = pd.read_csv("C:/Users/sonia/OneDrive/Desktop/Bot/metricheReddit/" + file, engine="c", quotechar='"',
                                 encoding='latin1', sep=";",
                                 usecols=[0], names=['Author'])
            testo = pd.read_csv("C:/Users/sonia/OneDrive/Desktop/Bot/metricheReddit/" + file, engine="c", quotechar='"',
                                encoding='latin1', sep=";", usecols=[1], names=['text'])
            author = autore['Author'].tolist()
            twet = testo['text'].tolist()

            print(line_count)

            for x in range(len(twet)):
                control = isinstance(twet[x], str)
                if not control:
                    twet[x] = "a"
                line_count += 1
                print(line_count)

                out_file.write(str(author[x]) + ',' + "\t")
                out_file.write(str(self.totalCharacter(twet[x])) + ',' + "\t")
                out_file.write(str(self.CharacterUpper(twet[x])) + ',' + "\t")
                out_file.write(str(self.CharacterLower(twet[x])) + ',' + "\t")
                out_file.write(str(self.CharacterSpecial(twet[x])) + ',' + "\t")
                out_file.write(str(self.NumberCount(twet[x])) + ',' + "\t")
                out_file.write(str(self.Whitespace(twet[x])) + ',' + "\t")
                out_file.write(str(self.NumberWord(twet[x])) + ',' + "\t")
                out_file.write(str(self.AvgLenWord(twet[x])) + ',' + "\t")
                out_file.write(str(self.NumberPhrase(twet[x])) + ',' + "\t")
                out_file.write(str(self.AvgLenPhrase(twet[x])) + ',' + "\t")
                out_file.write(str(self.PunctuationFrequency(twet[x])) + ',' + "\t")
                out_file.write(str(self.WordLower(twet[x])) + ',' + "\t")
                out_file.write(str(self.WordUpper(twet[x])) + ',' + "\t")
                out_file.write(str(self.VocabularyWealth(twet[x])) + ',' + "\t")
                out_file.write(str(self.UrlNumber(twet[x])) + ',' + "\t")
                out_file.write(str(self.Fkgl(twet[x])) + ',' + "\t")
                out_file.write(str(self.Fre(twet[x])) + ',' + "\t")
                out_file.write(str(self.Dcr(twet[x])) + ',' + "\t")
                out_file.write(str(self.Ari(twet[x])) + ',' + "\t")
                out_file.write(str(self.Cli(twet[x])) + ',' + "\t")
                out_file.write(str(self.Gf(twet[x])) + ',' + "\t")
                out_file.write(str(self.Smog(twet[x])) + ',' + "\t")
                out_file.write(str(self.Lw(twet[x])) + "\n")



inst = Metrics()  
nome = "chiai"
inst.CreateTable(nome)
