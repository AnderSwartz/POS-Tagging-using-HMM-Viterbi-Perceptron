import collections
from sys import prefix
import numpy as np

class Perceptron(object):
    """Base Perceptron model class"""

    def __init__(self, tag_types):
        self.weights = collections.Counter()
        self.tag_types = tag_types
        self.prefixes = collections.Counter()

    def get_features(self, prevTag, tag, word=None):
        '''Given a previous tag, current tag, and current word, generate a list of features.  Note that word may be omitted for <EOS>'''
        if word is not None:
            features_list = [(prevTag, tag),(word,tag)]
            #simple version of using prefixes
            # if(len(word)>8):
            #     features_list.append((word[0:3],tag))

            #complex version of checking prefixes
            #this involved finding the most frequently occuring prefixes using my
            #findRelaventPrefixesFromData function before training
            # if (word[0:3],tag) in self.prefixes:
                #     features_list.append((word[0:3],tag))
                # elif (word[0:2],tag) in self.prefixes:
                #     features_list.append((word[0:2],tag))

            #using both upper and lower case version if word isupper
            if(word[0].isupper() and prevTag == "<BOS>"):
                features_list.append((word.lower(),tag))
            #using length of word
            # features_list.append((len(word),tag))
        else:
            features_list = [(prevTag, tag)]
        # features_list.append((prevTag[0],tag))

        return features_list

    def score_features(self, features_list):
        '''Given a list of features, compute the score from those features'''
        score = 0
        for feat in features_list:
            score += self.weights[feat]
        return score

    def score_features2(self, features_list):
        '''Given a list of features, compute the score from those features'''
        score = 0
        for feat in features_list:
            score += self.weights[feat]
        return score    

    def train(self, sents, epochs=10):
        '''Given a list of sentences in the form:
        ['space separated string', ['O', 'O', 'O']]
        Train the perceptron for the given number of epochs'''
        for epoch in range(epochs):
            for sent, tags in sents:
                self.train_line(sent, tags)

    def train_line(self, sent, tags):
        '''Trains from a single sentence.  
        sent is a space separated string
        tags is a list of correct tags'''
        mytags = self.viterbi(sent)
        # mytags = "B-geo-loc"
        prevCorrect = '<BOS>'
        prevPred = '<BOS>'
        for w, c, p in zip(sent.split(' '), tags, mytags):
            if c != p:#if we guessed incorretly...
                for feat in self.get_features(prevCorrect, c, w):
                    self.weights[feat] += 1
                for feat in self.get_features(prevPred, p, w):
                    self.weights[feat] -= 1
            prevCorrect = c
            prevPred = p
        if c != p:
            # If the final tag is wrong, also update <EOS>
            for feat in self.get_features(c, '<EOS>'):
                self.weights[feat] += 1
            for feat in self.get_features(p, '<EOS>'):
                self.weights[feat] -= 1

    def tag_sents(self, sents, outFile='dev-percep.out'):
        '''Given a list of sentences in the form ['sample sentence here', ['O', 'O', 'O'] ], predicts tag sequence and writes to file for scoring. '''
        with open(outFile,'w') as g:
            for sent, tags in sents:
                mytags = self.viterbi(sent)
                for s, c, m in zip(sent.split(' '), tags, mytags):
                    g.write(s + ' ' + c + ' '+ m + '\n')
                g.write('\n')

    def viterbi(self, sent):#sent is a phrase
        '''Given a space separated string as input, produce the 
        highest weighted tag sequence for that string.'''
        # You will replace this, for now it just returns 'O' for every tag
        sent = sent.split()
        scores = np.full(shape=(len(tagTypes),len(sent)),fill_value=-10000000000,dtype=float)
        pointerMatrix = np.zeros(shape=(len(tagTypes),len(sent)),dtype=np.int32)

        for tag in range(len(tagTypes)):
            currentTagToCheck = tagTypes[tag]
            firstWord = sent[0]
            newScore = self.score_features(self.get_features("<BOS>",tagTypes[tag],firstWord))
            scores[tag,0] = newScore
        
        for m in range(1,len(sent)):
            currentWord = sent[m]
            for tag in range(len(tagTypes)):
                currentTagToCheck = tagTypes[tag]
                maxScoreForTagPossibility = -100000000
                for previousTagPossibilityIndex in range(len(tagTypes)): 
                    previousPossibleTag = tagTypes[previousTagPossibilityIndex]
                    
                    emission_and_transition_score = self.score_features(self.get_features(previousPossibleTag,tagTypes[tag],currentWord))
                    scoreFromPreviousTag = scores[previousTagPossibilityIndex][m-1] 
                    newScore = emission_and_transition_score + scoreFromPreviousTag
                
                    if(newScore>maxScoreForTagPossibility):
                        maxScoreForTagPossibility  = newScore
                        winningPointer = previousTagPossibilityIndex
                pointerMatrix[tag,m] = winningPointer
                scores[tag,m] = maxScoreForTagPossibility

        maxFinalScore = -1000000000
        for tag in range(len(tagTypes)):
            emission_and_transition_score = self.score_features(self.get_features(tagTypes[tag],"<EOS>"))
            scoreFromPreviousTag = scores[tag][-1]
            newScore = scoreFromPreviousTag+emission_and_transition_score
            if(newScore>maxFinalScore):
                maxFinalScore = newScore
                finalTagIndex = tag

        finalTagSequenceGuess = []
        predictedTagForM = finalTagIndex
        for m in range(len(sent)-1,-1,-1):
            finalTagSequenceGuess.insert(0,predictedTagForM)
            predictedTagForM = pointerMatrix[predictedTagForM,m] 
        
        finalTagSequenceNames = []
        for m in range(len(sent)):
            finalTagSequenceNames.append(tagTypes[finalTagSequenceGuess[m]])
        return finalTagSequenceNames

def setup(filename,encodings="utf8"):
    data = []
    newSentence = ""
    newTags = []
    totalTokens = 0
    wordTypes = {}
    tagTypes = {}
    for line in open(filename):
        if(line.strip()):
            tokens = line.split()
            newSentence+=(tokens[0]+" ")
            newTags.append(tokens[1])
            totalTokens+=1
            if(tokens[0] not in wordTypes):
                wordTypes[tokens[0]] = None 
            if(tokens[1] not in tagTypes):
                tagTypes[tokens[1]] = None
            # countPrefixes(tokens[0])
        else:
            # newTags.append("EOS")
            newItem = [newSentence,newTags] #try getting rid of :-1
            data.append(newItem)
            newSentence = ""
            newTags = []
    # print(data[0])
    # print(data[0])
    return data, totalTokens, wordTypes.keys(),tagTypes.keys()

def findRelaventPrefixesFromData(filename):
    # data = data.encode('utf8')  
    prefixes = collections.Counter()
    for line in open(filename,encoding="utf8"):
        if(line.strip()):
            tokens = line.split()
            word = tokens[0]
            word = word.lower()
            tag = tokens[1]
            if(len(word)>8):
                # prefixes[word[0:2]] = prefixes.get(word[0:2],0)+1
                # prefixes[word[0:3]] = prefixes.get(word[0:3],0)+1
                
                prefixes[(word[0:2])] +=1
                prefixes[(word[0:3])] +=1
    commonPrefixes = collections.Counter() 
    for key in prefixes:
        if prefixes[key] >10:
            commonPrefixes[key] = 1
        # if prefixes[(word[0:3],tag)] <3:
        #     del prefixes[(word[0:3],tag)]
    return commonPrefixes
    
trainingData, totalTokens, wordTypes, tagTypes  = setup("train")
devData, a, b, c  = setup("dev")
testData, a, b, c  = setup("test")


tagTypes = list(tagTypes)


perceptron = Perceptron(tagTypes)

#used when attempted to add prefixes as feature
# perceptron.prefixes = findRelaventPrefixesFromData("train")

perceptron.train(trainingData,7)
perceptron.tag_sents(devData)
# perceptron.tag_sents(testData)