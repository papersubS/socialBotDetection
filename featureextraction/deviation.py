import glob, os
import pandas as pd

filenumb = 0

with open('C:/Users/sonia/OneDrive/Desktop/Bot/metricheReddit/Devbot.csv', 'a', encoding='utf-8') as out_file:
    out_file.write("Author" + ',')
    out_file.write("NumberOfTotalCharactersStd" + ',')
    out_file.write("NumberOfUppercaseCharactersStd" + ',')
    out_file.write("NumberOfLowercaseCharactersStd" + ',')
    out_file.write("NumberOfSpecialCharactersStd" + ',')
    out_file.write("NumberOfEmoticonStd" + ',')
    out_file.write("NumberOfNumbersStd" + ',')
    out_file.write("NumberOfBlanksStd" + ',')
    out_file.write("NumberOfWordsStd" + ',')
    out_file.write("LengthOfWordsStd" + ',')
    out_file.write("NumberOfPropositionsStd" + ',')
    out_file.write("PropositionsLengthStd" + ',')
    out_file.write("NumberOfPunctuationCharactersStd" + ',')
    out_file.write("NumberOfLowercaseWordsStd" + ',')
    out_file.write("NumberOfUppercaseWordsStd" + ',')
    out_file.write("VocabularyRichnessStd" + ',')
    out_file.write("NumberOfURLsStd" + ',')
    out_file.write("FleschKincaidGradeLevelStd" + ',')  
    out_file.write("FleschReadingEaseStd" + ',')
    out_file.write("DaleChallReadabilityStd" + ',')
    out_file.write("AutomatedReadabilityIndexStd" + ',')
    out_file.write("ColemanLiauIndexStd" + ',')
    out_file.write("GunningFogStd" + ',')
    out_file.write("SMOG(SimpleMeasureOfGobbledygook)Std" + ',')
    out_file.write("LinsearWriteStd" + "\n")
    os.chdir('C:/Users/sonia/METRICHE/sp1/')
    frame = []
    for file in glob.glob("*"):
        print(file)
        filenumb += 1
        print("file:" + str(filenumb))
        df = pd.read_csv('C:/Users/sonia/OneDrive/Desktop/Bot/metricheReddit/bot/' + file, engine="python", quotechar='"',
                         encoding='latin1', sep=";", header=0)
        # a = df.groupby("Autore").std(ddof=0)
        frame.append(df)
    merged = pd.concat(frame, axis=0, ignore_index=True)


    g = merged.groupby(['Autore']).std(ddof=0)
    g.to_csv('C:/Users/sonia/OneDrive/Desktop/Bot/metricheReddit/Devbot.csv', mode='a', sep=',', header=False, index=1)
