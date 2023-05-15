import glob, os
import pandas as pd
from pandas import DataFrame

filenumb = 0

with open('C:/Users/sonia/OneDrive/Desktop/Bot/metricheReddit/mediatroll.csv', 'a', encoding='utf-8') as out_file:
    out_file.write("Author" + ',')
    out_file.write("NumberOfTotalCharactersAvg" + ',')
    out_file.write("NumberOfUppercaseCharactersAvg" + ',')
    out_file.write("NumberOfLowercaseCharactersAvg" + ',')
    out_file.write("NumberOfSpecialCharactersAvg" + ',')
    out_file.write("NumberOfEmoticonAvg" + ',')
    out_file.write("NumberOfNumbersAvg" + ',')
    out_file.write("NumberOfBlanksAvg" + ',')
    out_file.write("NumberOfWordsAvg" + ',')
    out_file.write("LengthOfWordsAvg" + ',')
    out_file.write("NumberOfPropositionsAvg" + ',')
    out_file.write("PropositionsLengthAvg" + ',')
    out_file.write("NumberOfPunctuationCharactersAvg" + ',')
    out_file.write("NumberOfLowercaseWordsAvg" + ',')
    out_file.write("NumberOfUppercaseWordsAvg" + ',')
    out_file.write("VocabularyRichnessAvg" + ',')
    out_file.write("NumberOfURLsAvg" + ',')
    out_file.write("FleschKincaidGradeLevelAvg" + ',')  
    out_file.write("FleschReadingEaseAvg" + ',')
    out_file.write("DaleChallReadabilityAvg" + ',')
    out_file.write("AutomatedReadabilityIndexAvg" + ',')
    out_file.write("ColemanLiauIndexAvg" + ',')
    out_file.write("GunningFogAvg" + ',')
    out_file.write("SMOG(SimpleMeasureOfGobbledygook)Avg" + ',')
    out_file.write("LinsearWriteAvg" + "\n")

    os.chdir("C:/Users/sonia/OneDrive/Desktop/Bot/metricheReddit/troll/")
    frame = []
    for f in glob.glob("*"):
        filenumb += 1
        print("file:" + str(filenumb))
        df = pd.read_csv('C:/Users/sonia/OneDrive/Desktop/Bot/metricheReddit/troll/' + f, engine="python", quotechar='"',
                         encoding='latin1', sep=";", header=0)
        # a = df.groupby("Autore").std(ddof=0)
        frame.append(df)
    merged = (pd.concat(frame, axis=0, ignore_index=True))
    print(merged['Autore'])
    a = merged.groupby('Autore').mean()
    print(a)
    a.to_csv('C:/Users/sonia/OneDrive/Desktop/Bot/metricheReddit/mediatroll.csv', mode='a', sep=',', header=False, index=1)
   
