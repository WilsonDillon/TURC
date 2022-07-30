import pandas as pd
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

def GetData(fileName, sheetIndex):
    xls = pd.ExcelFile(fileName)
    sheetNames = xls.sheet_names
    return pd.read_excel(xls, sheet_name=sheetNames[sheetIndex], index_col=2), sheetNames

def StartARIMAForecasting(Actual, P, D, Q):
    model = ARIMA(Actual, order=(P,D,Q))
    model_fit = model.fit()
    prediction = model_fit.forecast()[0]
    return prediction

for x in range(0,3):
    timeSeries = GetData('model_data.xlsx', x)[0]
    timeSeries = timeSeries.loc[:, timeSeries.columns!='IF']
    dataName = GetData('model_data.xlsx', x)[1][x] + " Data:"
    print(dataName)

    numberOfElements = len(timeSeries)
    trainingSet = [0.60, 0.70, 0.80, 0.90]
    for y in range(0,4):
        trainingSize = int(numberOfElements * trainingSet[y])
        trainingData = timeSeries[0:trainingSize]
        testData = timeSeries[trainingSize:numberOfElements]
        predictionsIndex = [x for x in timeSeries.index][trainingSize:numberOfElements]

        actual = [x for x in timeSeries.loc[:,"FN"]]
        predictions = list()

        for timepoint in range(len(testData)):
            actualValue = [x for x in testData.loc[:,"FN"]][timepoint]
            prediction = StartARIMAForecasting(actual, 1,0,0)
            predictions.append(prediction)
            actual.append(actualValue)

        print("  Training Factor: ",trainingSet[y])
        tmp = predictions[0]
        for x in range(len(predictions)):
            if (x == len(predictions)-1):
                predictions[x] = tmp
            else: predictions[x] = predictions[x+1]
            print('    Actual=%f, Predicted=%f' % (actual[trainingSize:numberOfElements][x], predictions[x]))

        predictionsDF = pd.DataFrame(predictions, predictionsIndex)

        error = mean_squared_error(testData, predictions)
        print('  Test Mean Squared Error: %.3f' % error)
        print()

        chartTitle = str(dataName) + " Training Factor: " + str(trainingSet[y])
        pyplot.title(chartTitle)
        pyplot.xlabel('FT')
        pyplot.ylabel('FN')
        pyplot.plot(testData)
        pyplot.plot(predictionsDF, color='red')
        pyplot.show()