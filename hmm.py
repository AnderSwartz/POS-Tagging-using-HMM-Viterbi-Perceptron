import numpy as np

emissionWeightsNumerator = {}
emissionWeightsDenominator = {}
transitionWeightsNumerator = {}
transitionWeightsDenominator = {}

def setup(filename):
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
        else:
            newItem = [newSentence,newTags]
            data.append(newItem)
            newSentence = ""
            newTags = []

    return data, totalTokens, wordTypes.keys(),tagTypes.keys()

def setProbabilities(data):
    for item in data:
        sentence = item[0].split()
        tags = item[1]
        previousTag = "BOS"
        for m in range(len(sentence)):
            currentWord = sentence[m]
            currentTag = tags[m]
            emissionRFE(currentWord,currentTag)
            transitionRFE(currentTag,previousTag)
            previousTag = currentTag
        #we only looped through the length of the sentence but there is an extra EOS tag that need to be handled
        currentTag="EOS"
        transitionRFE(currentTag,previousTag)
    # return emissionWeightsNumerator,emissionWeightsDenominator

def emissionRFE(currentWord,currentTag):
    emissionWeightsNumerator[currentWord,currentTag] = emissionWeightsNumerator.get((currentWord,currentTag),0)+1
    emissionWeightsDenominator[currentTag] = emissionWeightsDenominator.get(currentTag,0)+1

def transitionRFE(currentTag,previousTag):
        transitionWeightsNumerator[currentTag,previousTag] = transitionWeightsNumerator.get((currentTag,previousTag),0)+1
        transitionWeightsDenominator[previousTag] = transitionWeightsDenominator.get(previousTag,0)+1

def getEmissionProbability(word,tagGuess):
    if(word not in wordTypes):
        word = "<UNK>"
    #best is .1 and .01 and diviving by len(wordTypes)
    #to find first set of emission probs that i listed in 2.2, use this:
    # return ((emissionWeightsNumerator.get((word,tagGuess),0)+.1)/(emissionWeightsDenominator[tagGuess]))

    return np.log((emissionWeightsNumerator.get((word,tagGuess),0)+.01)/(emissionWeightsDenominator[tagGuess]+(len(wordTypes))))

def getTransitionProbability(currentTagGuess,previousTag):
    return np.log((transitionWeightsNumerator.get((currentTagGuess,previousTag),0))/transitionWeightsDenominator[previousTag])
    
def viterbi(phrase):
    #each score represents the total of the emission, transistion, 
    # and previous score from the optimal previous tag to come from
    scores = np.full(shape=(len(tagTypes),len(phrase)),fill_value=-10000000000,dtype=float)
    #the pointer matrix has the same dimensions as scores, 
    # but contains the index for the tag that results in the optimal score for the location
    pointerMatrix = np.zeros(shape=(len(tagTypes),len(phrase)),dtype=np.int32)

    #handing the first word, only using the emission and transition score
    for tag in range(len(tagTypes)):
        currentTagToCheck = tagTypes[tag]
        emissionProbability = getEmissionProbability(phrase[0],currentTagToCheck)
        transitionProbability = getTransitionProbability(currentTagToCheck,"BOS")
        newScore = (emissionProbability+transitionProbability)
        scores[tag,0] = newScore
    
    #handling the rest of the words, using emission, transition, 
    # and score from previous tag that is optimal to come from
    for m in range(1,len(phrase)):
        currentWord = phrase[m]
        for tag in range(len(tagTypes)):
            currentTagToCheck = tagTypes[tag]
            #emission probability only needs to be found once for each tag and word combination
            emissionProbability = getEmissionProbability(currentWord,currentTagToCheck)
            maxScoreForTagPossibility = -100000000
            for previousTagPossibilityIndex in range(len(tagTypes)): 

                previousPossibleTag = tagTypes[previousTagPossibilityIndex]
                #contrastly, the transition probability needs to checked for all previous tags to find the optimal one
                transitionProbability = getTransitionProbability(currentTagToCheck,previousPossibleTag)
                scoreFromPreviousTag = scores[previousTagPossibilityIndex][m-1]
               
                newScore = emissionProbability+transitionProbability+scoreFromPreviousTag
                #if a transition from a certain previous tag to the current gives the optimal score,
                #keep track of it
                if(newScore>maxScoreForTagPossibility):
                    maxScoreForTagPossibility  = newScore
                    winningPointer = previousTagPossibilityIndex
            
            #after all words are processes, the "winning pointer" is the optimal previous tag
            #that gives the greatest probability for the last word
            pointerMatrix[tag,m] = winningPointer
            scores[tag,m] = maxScoreForTagPossibility

    #finally, the last transition to "EOS" is handling only using the 
    # transition probability and max score for each tag       
    maxFinalScore = -1000000000
    for tag in range(len(tagTypes)):
        currentTagToCheck = tagTypes[tag]
        transitionProbability = getTransitionProbability("EOS",currentTagToCheck)
        scoreFromPreviousTag = scores[tag][-1]
        newScore = scoreFromPreviousTag+transitionProbability
        if(newScore>maxFinalScore):
            maxFinalScore = newScore
            finalTagIndex = tag

    finalTagSequenceGuess = []
    predictedTagForM = finalTagIndex
    #tracing back optimal tax sequence using pointer matrix
    for m in range(len(phrase)-1,-1,-1):
        finalTagSequenceGuess.insert(0,predictedTagForM)
        predictedTagForM = pointerMatrix[predictedTagForM,m] 

    #converting answer from tag indices to tags
    finalTagSequenceNames = []
    for m in range(len(phrase)):
        finalTagSequenceNames.append(tagTypes[finalTagSequenceGuess[m]])
    return finalTagSequenceNames
    
def testOnFile(testingData):
    devOut = open("dev.out","a")
    # devOut = open('dev2.txt', 'a')
    devOut.truncate(0)
    for item in testingData:
        phrase = item[0].split()
        correctTags = item[1]
        finalTagSequenceGuesses= viterbi(phrase)
        for m in range(len(phrase)):
            currentWord = phrase[m]
            correctTag = correctTags[m]
            guessedTag = finalTagSequenceGuesses[m]
            lineToWrite = currentWord+" " +correctTag+" "+guessedTag+"\n"
            devOut.write(lineToWrite)
        devOut.write("\n")

trainingData, totalTokens, wordTypes, tagTypes  = setup("train")
tagTypes = list(tagTypes)
print(totalTokens)
print(len(wordTypes))
print(len(tagTypes))

# a b and c are unused variables
#  (I'm still learning Python and I know this isn't the most efficient)

setProbabilities(trainingData)



#2.1
# print(getTransitionProbability("B-person","O"))
# print(getTransitionProbability("B-person","B-person"))
# print(getTransitionProbability("I-person","B-person"))
# print(getTransitionProbability("B-person","I-person"))
# print(getTransitionProbability("I-person","I-person"))
# print(getTransitionProbability("O","I-person"))
print()
#2.2
# print(getEmissionProbability("God","B-person"))
# print(getEmissionProbability("God","O"))
# print(getEmissionProbability("Lindsay","B-person"))
# print(getEmissionProbability("Lindsay","O"))

#2.3
devData, a, b, c = setup("dev") 
testOnFile(devData)

#emmision probability = number of instances of this word and this tag, 
# divided by number of occurences of this tag

# transition probability = number of instances of this tag guess followed
#  by the previous tag, divided by total instances of the previous tag