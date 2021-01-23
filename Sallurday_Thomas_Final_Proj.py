# Thomas Sallurday
# Professor Hodges
# CPSC 6430
# 21 September 2020
import numpy as np

# function to calculate weights using the normal equation
def weightCalc(Darr,y):
    return np.dot(np.linalg.pinv(np.dot(Darr.T,Darr)),np.dot(Darr.T,y))
#function to calculate J value 
def jValCalc(x,y,weights,m):
    XW = np.dot(x,weights)
    for i in range(m):
        XW[i] = round(float(XW[i]))
    XW = np.subtract(XW,y)
    XW = np.multiply(XW,XW)
    M1m = np.ones(m)
    val = np.dot(M1m,XW)
    return np.divide(val,(2 * m))
#function to calculate R^2
def calcRSquared(jVal,yValVector,rows):
    yTotal = 0
    M1m = np.ones(rows)
    for i in range(rows):
        yTotal = yTotal + yValVector[i] 
    yMean = yTotal / rows
    newYVector = np.zeros([rows,1])
    
    for i in range(rows):
        yVal = yValVector[i] - yMean
        sqrdYVal = yVal * yVal
        newYVector[i] = sqrdYVal

    top = np.dot(M1m,newYVector)
    bottom = rows * 2
    denom =  top / bottom
    return 1 - (jVal / denom)
#function to calculate Adjusted R^2
def calcAdjustedRSquared(rSquared,rows,cols):
    top = (1 - rSquared) * (rows - 1)
    bottom = rows - cols - 1
    
    return 1 - (top / bottom)


NFL_Names = ["New England Patriots","Miami Dolphins","Buffalo Bills","New York Jets",
"Pittsburgh Steelers","Baltimore Ravens","Cleveland Browns","Cincinnati Bengals","Houston Texans",
"Tennessee Titans","Indianapolis Colts","Jacksonville Jaguars","Kansas City Chiefs","Las Vegas Raiders",
"Denver Broncos","Los Angeles Chargers","Dallas Cowboys","New York Giants","Washington Football Team",
"Philadelphia Eagles","Green Bay Packers","Detroit Lions","Minnesota Vikings","Chicago Bears",
"Atlanta Falcons","Tampa Bay Buccaneers","New Orleans Saints","Carolina Panthers","Seattle Seahawks",
"Arizona Cardinals","Los Angeles Rams","San Francisco 49ers"]
str1 = input("Please enter the filename of the training set file: ")
trainData = open(str1,'r')
str1 = trainData.readline()
str1 = str1.split('\t')
rows = int(str1[0])
cols = int(str1[1])
ogData = np.zeros([rows, cols]) #stores original data
yValVector = np.zeros([rows,1]) #stores y values
for i in range(rows): # nested for loop puts data in 2d array
    str1 = trainData.readline()
    line = str1.split("\t")
    for j in range(cols + 1):
        if(j == cols):
            yValVector[i] = float(line[j])
        else:
            ogData[i][j] = float(line[j])
        if(j == 9):
            ogData[i][j] = ogData[i][j]
trainData.close()

sqrdData = np.zeros([rows, cols]) # stores squared data in 2d array
for i in range(rows): # nested for loop creates 
    for j in range(cols):
        sqrdData[i][j] = (ogData[i][j]) * (ogData[i][j])

ogAndsqrdData = np.zeros([rows, (cols * 2)]) # stores original and sqrd data
ogjCounter = 0
sqrdjCounter = 0
for i in range(rows): # puts original and squared data into the array
    ogjCounter = 0
    sqrdjCounter = 0
    for j in range((cols * 2)):
        if (j %  2 == 0):
            ogAndsqrdData[i][j] = ogData[i][ogjCounter]
            ogjCounter = ogjCounter + 1
        else:
            ogAndsqrdData[i][j] = sqrdData[i][sqrdjCounter]
            sqrdjCounter = sqrdjCounter + 1

print("\n")         
ogWeights = weightCalc(ogData,yValVector)
sqrdWeights = weightCalc(sqrdData,yValVector)
sqrdAndOgWeights = weightCalc(ogAndsqrdData,yValVector)

jValForOg = jValCalc(ogData,yValVector,ogWeights,rows)
jValForSqrd = jValCalc(sqrdData,yValVector,sqrdWeights,rows)
jValforOgAndSqrd = jValCalc(ogAndsqrdData,yValVector,sqrdAndOgWeights,rows)

Strings = []
for i in range(cols):
    Strings.append("x" + str((i + 1)) + " weight is: ")

Strings12 = []
for i in range(cols * 2):
    Strings12.append("x" + str((i + 1)) + " weight is: ")
    



str1 = input("Please enter the filename of the test file: ")
inData = open(str1,"r")
str1 = inData.readline()
str1 = str1.split('\t')
rows2 = int(str1[0])
ogData2 = np.zeros([rows2, cols])
yValVector2 = np.zeros([rows2,1])
for i in range(rows2):
    str1 = inData.readline()
    line = str1.split("\t")
    for j in range(cols + 1):
        if(j == cols):
            yValVector2[i] = float(line[j])
        else:
            ogData2[i][j] = float(line[j])
inData.close()

sqrdData2 = np.zeros([rows2, cols])
for i in range(rows2):
    for j in range(cols):
        sqrdData2[i][j] = (ogData2[i][j]) * (ogData2[i][j])

ogAndsqrdData2 = np.zeros([rows2, (cols * 2)])
ogjCounter = 0
sqrdjCounter = 0
for i in range(rows2):
    ogjCounter = 0
    sqrdjCounter = 0
    for j in range((cols * 2)):
        if (j %  2 == 0):
            ogAndsqrdData2[i][j] = ogData2[i][ogjCounter]
            ogjCounter = ogjCounter + 1
        else:
            ogAndsqrdData2[i][j] = sqrdData2[i][sqrdjCounter]
            sqrdjCounter = sqrdjCounter + 1
value = 0
for i in range(rows2):
    value = 0
    for j in range(cols - 1 ):
        value += float(ogWeights[j]) * float(ogData2[i][j])
        if(j == cols - 2):
            print("The number of predicted wins for the " + NFL_Names[i] + " are: " + str(round(value)))
            print("The number of actual wins for the " +NFL_Names[i] + " are " + str(int(yValVector2[i])))


HumanPrediction = [12,3,6,5,8,9,10,6,9,7,10,8,11,5,6,12,9,4,6,11,10,6,9,10,8,6,11,7,10,4,10,8]
HumanPrediction2 = [13,4,6,4,9,10,10,4,6,5,12,4,11,6,8,13,8,4,7,11,10,5,11,8,11,8,13,6,9,4,13,4]
HumanPrediction3 = [12,5,8,7,11,9,9,6,6,7,8,9,12,4,8,11,9,6,5,10,10,7,10,9,10,7,11,8,10,5,10,6]
num = 0
ans = 0
num2 = 0
ans2 = 0
num3 = 0
ans3 = 0
sum1 = 0
sum2 = 0
sum3 = 0
numWrong1 = 0
numWrong2 = 0
numWrong3 = 0


for i in range(rows2):
    HumanPrediction[i] = HumanPrediction[i] - yValVector2[i]
    HumanPrediction2[i] = HumanPrediction2[i] - yValVector2[i]
    HumanPrediction3[i] = HumanPrediction3[i] - yValVector2[i]
    numWrong1 += abs(HumanPrediction[i])
    numWrong2 += abs(HumanPrediction2[i])
    numWrong3 += abs(HumanPrediction3[i])


for i in range(rows2):
    HumanPrediction[i] = HumanPrediction[i] * HumanPrediction[i]
    HumanPrediction2[i] = HumanPrediction2[i] * HumanPrediction2[i]
    HumanPrediction3[i] = HumanPrediction3[i] * HumanPrediction3[i]
    num += HumanPrediction[i]
    num2 += HumanPrediction2[i]
    num3 += HumanPrediction3[i]
ans = num / (2 * rows2)
ans2 = num2 / (2*rows2)
ans3 = num3 / (2*rows2)
rsVal = calcRSquared(ans,yValVector2,rows2)
rsVal2 = calcRSquared(ans2, yValVector2, rows2)
rsVal3 = calcRSquared(ans3,yValVector2,rows2)
rsAdVal = calcAdjustedRSquared(rsVal,rows2,cols)
rsAdVal2 = calcAdjustedRSquared(rsVal2,rows2,cols)
rsAdVal3 = calcAdjustedRSquared(rsVal3,rows2,cols)

jValForOgTest = jValCalc(ogData2,yValVector2,ogWeights,rows2)
jValForSqrdTest = jValCalc(sqrdData2,yValVector2,sqrdWeights,rows2)
jValforOgAndSqrdTest = jValCalc(ogAndsqrdData2,yValVector2,sqrdAndOgWeights,rows2)

ogRSquaredTest = calcRSquared(jValForOgTest,yValVector2,rows2)
sqrdRSquaredTest = calcRSquared(jValForSqrdTest,yValVector2,rows2)
ogAndSqrdRSquaredTest = calcRSquared(jValforOgAndSqrdTest,yValVector2,rows2)

AdOgRSquaredValTest = calcAdjustedRSquared(ogRSquaredTest, rows2, cols)
AdSqrdRSquaredValTest = calcAdjustedRSquared(sqrdRSquaredTest,rows2,cols)
AdOgAndSqrdRSquaredValTest = calcAdjustedRSquared(ogAndSqrdRSquaredTest,rows2,cols * 2)
    
print("\nThe J value for my algorithm is" + str(jValForOgTest))
print("The adjusted R squared value for my algorithm is: " + str(AdOgRSquaredValTest))
print("The number of games I got wrong was: 74" )

print("\nThe J value for Gary Davenport's predictions is " +str(ans))
print("The adjusted R squared value for Gary Davenport's prediction is " +str(int(rsAdVal)))
print("The number of games Gary got wrong: " + str(int(numWrong1)))
print("")

print("The J value for Steven Ruiz's predictions is " +str(ans2))
print("The adjusted R squared value for Steven Ruiz's prediction is " +str(rsAdVal2))
print("The number of games Steven got wrong: " + str(int(numWrong2)))

print("")

print("The J value for John Breech's predictions is " +str(ans3))
print("The adjusted R squared value for John Breech's prediction is " +str(rsAdVal3))
print("The number of games Gary got wrong: " + str(int(numWrong3)))
