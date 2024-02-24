from math import log2
import numpy as np

def setup(filename):
    data = []
    newSentence = ""
    newTags = []
    totalTokens = 0
    wordTypes = {}
    tagTypes = {}
    for line in open(filename):
        if(line.strip()):
        # print(line)
            tokens = line.split()
            newSentence+=(tokens[0]+" ")
            newTags.append(tokens[1])
            totalTokens+=1
            if(tokens[0] not in wordTypes):
                wordTypes[tokens[0]] = None 
            if(tokens[1] not in tagTypes):
                tagTypes[tokens[1]] = None
        else:
            # newTags.append("EOS")
            newItem = [newSentence[:-1],newTags]
            data.append(newItem)
            newSentence = ""
            newTags = []
    # print(data[0])
    # print(data[0])
    return data, totalTokens, wordTypes.keys(),tagTypes.keys()
    # print(data)
    # print(totalTokens)
    # print(len(wordTypes.keys()))
    # print(len(tagTypes.keys()))
    #did you get 46469, 10586 and 21 for total tokens, word types and tag types

# transitionWeights = np.zeros(wordTypes*len(tagTypes))
emissionWeightsNumerator = {}
emissionWeightsDenominator = {}
transitionWeightsNumerator = {}
transitionWeightsDenominator = {}
# transitionWeights = np.zeros(wordTypes*len(tagTypes))
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
        # thing = transitionWeightsNumerator.get(("O","O"),0)
        transitionWeightsNumerator[currentTag,previousTag] = transitionWeightsNumerator.get((currentTag,previousTag),0)+1
        transitionWeightsDenominator[previousTag] = transitionWeightsDenominator.get(previousTag,0)+1

def getEmissionProbability(word,tagGuess):
    #three possibilits
    # if(emissionWeightsNumerator.get((word,tagGuess),0)!=None):
    if(word not in wordTypes):
        word = "<UNK>"
    # print(word)
    return ((emissionWeightsNumerator.get((word,tagGuess),0)+.1)/emissionWeightsDenominator[tagGuess])

def getTransitionProbability(currentTagGuess,previousTag):
    # try:
        return ((transitionWeightsNumerator.get((currentTagGuess,previousTag),0)+.1)/transitionWeightsDenominator[previousTag])
    # except: 
    # try:
    #     return ((transitionWeightsNumerator[currentTagGuess,previousTag]+.1)/transitionWeightsDenominator[previousTag])
    # except:
    #     return 0

def viterbi(phrase,tags):
    print(len(tagTypes))
    print(len(phrase))
    scores = np.full(shape=(len(tagTypes),len(phrase)),fill_value=-100000)#rows = tagTypes, cols = words in phrase
    pointerMatrix = np.zeros(shape=(len(tagTypes),len(phrase)))
    previousTag = "BOS"

    #handle first word different because its always coming from "BOS"
    for tag in range(len(tagTypes)):
        currentTagToCheck = tagTypes[tag]
        emissionProbability = getEmissionProbability(phrase[0],currentTagToCheck)
        transitionProbability = getTransitionProbability(currentTagToCheck,"BOS")
        newScore = emissionProbability+transitionProbability
        # if(newScore>scores[tag][0]):
        scores[tag][0] = newScore
    print(scores[:,0])
    print()
    print("the best tag for word:",phrase[0],"is",tagTypes[np.argmax(scores[:,0])])

    

    for m in range(1,len(phrase)):
        currentWord = phrase[m]
        for tag in range(len(tagTypes)):
            currentTagToCheck = tagTypes[tag]
            emissionProbability = getEmissionProbability(currentWord,currentTagToCheck)

            for previousTagPossibilityIndex in range(len(tagTypes)): 
                # newScore = (getEmissionProbability(phrase[m],tagTypes[tag])+
                # getTransitionProbability(tagTypes[tag],previousTagPossibility)+
                # scores[tag][previousTagPossibility])
                previousPossibleTag = tagTypes[previousTagPossibilityIndex]

                transitionProbability = getTransitionProbability(currentTagToCheck,previousPossibleTag)
                scoreFromPreviousTag = scores[previousTagPossibilityIndex][m-1]
               
                newScore = emissionProbability+transitionProbability+scoreFromPreviousTag
                # print(scores[tag][m])
                if(newScore>scores[tag][m]):
                    scores[tag][m] = newScore
                    # pointerMatrix[tag][m] = previousTagPossibilityIndex
                    winningPointer = previousTagPossibilityIndex
                    # print("found a new high coming from",tagTypes[previousTagPossibilityIndex])
            # print()
            print("when assuming tag",currentTagToCheck,"it made the most sense to come from",tagTypes[winningPointer])
        print(scores[:,m])
        print("the best tag for word:",currentWord,"is",tagTypes[np.argmin(scores[:,m])])
        print()

    
    #after processing the entire sentence, just use transitionWeights for the EOS token
    # print(scores[:,-1])
    maxScore = 0
    for tag in range(len(tagTypes)):
        currentTagToCheck = tagTypes[tag]
        # emissionProbability = getEmissionProbability(phrase[0],currentTagToCheck)
        transitionProbability = getTransitionProbability("EOS",currentTagToCheck)
        scoreFromPreviousTag = scores[tag][-1]
        newScore = scoreFromPreviousTag+transitionProbability
        if(newScore>maxScore):
            maxScore = newScore
            finalTagIndex = tag  
    #CURRENTLY NOT SETTING FINALTAGINDEX TO ANYTHING (ALL SCORES ARE 0?)
    print(tagTypes[finalTagIndex])
    print()
    print(pointerMatrix)
    print()
    #get pointers
    # currentTag= finalTagIndex
    # tagSequence = [finalTagIndex]
    # for i in range(len(phrase),0,-1):
    #     pointerMatrix[]

    


    # previousTag = "BOS"
    #     for m in range(len(sentence)):
    #         currentWord = sentence[m]
    #         for tag in tagTypes:

    #         previousTag = tags[m]
    # first_key = list(colors)[0]
    # first_val = list(colors.values())[0]

def testOnFile(testingData):
    # devOut = open("dev.out","a")
    devOut = open('dev2.txt', 'a')
    devOut.truncate(0)
    previousTag = "BOS"
    for item in testingData:
        sentence = item[0].split()
        tags = item[1]
        viterbi(sentence,tags)
        # previousTag = "BOS"
        # for m in range(len(sentence)):
        #     currentWord = sentence[m]
        #     for tag in tagTypes:

        #     previousTag = tags[m]
        
        
        #we only looped through the length of the sentence but there is an extra EOS tag that need to be handled
        # currentTag="EOS"
        # transitionRFE(currentTag,previousTag)


        # if(line.strip()):
        #     tokens = line.split()
        #     currentWord=tokens[0]
        #     guess = viterbi(currentWord,previousTag)
        #     previousTag = tokens[1]
        #     # print()
        #     lineToWrite = currentWord+" " +previousTag+" "+guess+"\n"
        #     devOut.write(lineToWrite)
        # else:
        #     previousTag = "BOS"
        #     devOut.write("\n")

    
trainingData, totalTokens, wordTypes, tagTypes  = setup("train")
tagTypes = list(tagTypes)
print(tagTypes)
testingData, a, b, c = setup("dev")
setProbabilities(trainingData)
# print(trainingData[0])
# print(testingData[0])
# print(trainingData[0])
# testOnFile(testingData)
# print(getEmissionProbability("STOP","O"))


def practiceViterbi(phrase):
    phrase = phrase.split()  
    scores = np.full(shape=(len(tagTypes),len(phrase)),fill_value=-100000,dtype=float)

    for tag in range(len(tagTypes)):
        currentTagToCheck = tagTypes[tag]
        emissionProbability = getEmissionProbability(phrase[0],currentTagToCheck)
        transitionProbability = getTransitionProbability(currentTagToCheck,"BOS")
        newScore = np.log2(emissionProbability*transitionProbability)
        # if(newScore>scores[tag][0]):
        scores[tag,0] = newScore
    print(scores)
    print("the best tag for word:",phrase[0],"is",tagTypes[np.argmax(scores[:,0])])

    for m in range(1,len(phrase)):
        currentWord = phrase[m]
        for tag in range(len(tagTypes)):
            currentTagToCheck = tagTypes[tag]
            emissionProbability = getEmissionProbability(currentWord,currentTagToCheck)
            maxScoreForTagPossibility = -10000
            for previousTagPossibilityIndex in range(len(tagTypes)): 
                # newScore = (getEmissionProbability(phrase[m],tagTypes[tag])+
                # getTransitionProbability(tagTypes[tag],previousTagPossibility)+
                # scores[tag][previousTagPossibility])
                previousPossibleTag = tagTypes[previousTagPossibilityIndex]

                transitionProbability = getTransitionProbability(currentTagToCheck,previousPossibleTag)
                scoreFromPreviousTag = scores[previousTagPossibilityIndex][m-1]
               
                newScore = np.log2(emissionProbability*transitionProbability)*scoreFromPreviousTag
                # print(scores[tag][m])
                if(newScore>maxScoreForTagPossibility):
                    maxScoreForTagPossibility  = newScore
                    # pointerMatrix[tag][m] = previousTagPossibilityIndex
                    winningPointer = previousTagPossibilityIndex
                    # print("found a new high coming from",tagTypes[previousTagPossibilityIndex])
            # print()
            scores[tag,m] = maxScoreForTagPossibility
            print("when assuming tag",currentTagToCheck,"it made the most sense to come from",tagTypes[winningPointer])
        print(scores)
        print("the best tag for word:",currentWord,"is",tagTypes[np.argmax(scores[:,m])])
        print()

    # maxFinalScore = -10000
    # for tag in range(len(tagTypes)):
    #     currentTagToCheck = tagTypes[tag]
    #     # emissionProbability = getEmissionProbability(phrase[0],currentTagToCheck)
    #     transitionProbability = getTransitionProbability("EOS",currentTagToCheck)
    #     scoreFromPreviousTag = scores[tag][-1]
    #     newScore = scoreFromPreviousTag+transitionProbability
    #     if(newScore>maxFinalScore):
    #         maxFinalScore = newScore
    #         finalTagIndex = tag
    # print("final tag = ",tagTypes[finalTagIndex])



    

# practiceViterbi("NYC is good")

# print(getTransitionProbability("O","O"))
# print(getTransitionProbability("I-other","O"))

# print(getTransitionProbability("O","I-other"))
# print(getTransitionProbability("O","O"))

# one = getTransitionProbability("O","O")
# two = getTransitionProbability("I-other","O")




# emissionWeightsNumerator,emissionWeightsDenominator = transitionProbabilities(data)

# print(getEmissionProbability("GA","B-geo-loc"))



# print(getTransitionProbability("B-person","O"))
# print(getTransitionProbability("B-person","B-person"))
# print(getTransitionProbability("I-person","B-person"))
# print(getTransitionProbability("B-person","I-person"))
# print(getTransitionProbability("I-person","I-person"))
# print(getTransitionProbability("O","I-person"))


# print(getEmissionProbability("God","B-person"))
# print(getEmissionProbability("God","O"))
# print(getEmissionProbability("Lindsay","B-person"))
# print(getEmissionProbability("Lindsay","O"))
# print(viterbi("charlene","B-person"))
# print(viterbi("town","B-movie"))



# print(transitionWeightsNumerator["O","I-person"])
#emmision probability = number of instances of this word and this tag, divided by number of occurences of this tag

# transition probability = number of instances of this tag guess followed by the previous tag, divided by total instances of the previous tag
