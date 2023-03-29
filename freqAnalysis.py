import spacy
import textacy
import json
from collections import Counter
from PyPDF2 import PdfReader

nlp = spacy.load('en_core_web_lg')

def freqAnalyser(noOfDocuments):
    verb_phrases = [] 
    nounChunk = []
    text=""
    patterns = [[{"POS": "VERB"}, {"POS": "ADV"}], [{"POS": "ADV"}, {"POS": "VERB"}]]
    for i in range(1,noOfDocuments+1):
        reader = PdfReader(f'data/{i}.pdf') #saved documents in a seperate data folder
        for page in reader.pages:
            text += page.extract_text()
    doc = nlp(text)
    verb_phrases.extend(list(textacy.extract.token_matches(doc, patterns=patterns))) #collecting verb phrases
    nounChunk.extend(list(doc.noun_chunks)) #collecting noun phrases
    verbFreq = Counter([str(verb_phrase).lower() for verb_phrase in verb_phrases])
    nounFreq = Counter([str(noun).lower() for noun in nounChunk])
    sortedVerbFreq = dict(sorted(verbFreq.items(), key=lambda x:x[1], reverse=True)) #sorting in descending order
    sortedNounFreq = dict(sorted(nounFreq.items(), key=lambda x:x[1], reverse=True))

    with open("verbList.json", "w") as outfile:
        json.dump(sortedVerbFreq, outfile)

    with open("nounList.json", "w") as outfile:
        json.dump(sortedNounFreq, outfile)

if __name__ == "__main__":
    freqAnalyser(10) #no.of documents to be analyzed