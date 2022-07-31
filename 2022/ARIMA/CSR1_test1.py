import pandas as pd
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import xlsxwriter

def GetData(fileName, sheetIndex):
    xls = pd.ExcelFile(fileName)
    sheetNames = xls.sheet_names
    return pd.read_excel(xls, sheet_name=sheetNames[sheetIndex], index_col=1), sheetNames

def StartARIMAForecasting(Actual, P, D, Q):
    model = ARIMA(Actual, order=(P,D,Q))
    model_fit = model.fit()
    prediction = model_fit.forecast()[0]
    return prediction

path = 'CSRDataResults.xlsx'
with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
    for x in range(0,2):
        timeSeries = GetData('CSRData.xlsx', x)[0]
        dataName = GetData('CSRData.xlsx', x)[1][x] + " Data"
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

            resultsDF = pd.DataFrame(columns = ['Actual', 'Predicted'])
            print("  Training Factor: ",trainingSet[y])
            tmp = predictions[0]
            for x in range(len(predictions)):
                if (x == len(predictions)-1):
                    predictions[x] = tmp
                else: predictions[x] = predictions[x+1]
                # print('    Actual=%f, Predicted=%f' % (actual[trainingSize:numberOfElements][x], predictions[x]))
                new_row = pd.DataFrame([[actual[trainingSize:numberOfElements][x], predictions[x]]], columns=['Actual', 'Predicted'])
                resultsDF = pd.concat([resultsDF, new_row], ignore_index=True)

            predictionsDF = pd.DataFrame(predictions, predictionsIndex)

            error = pd.DataFrame([[mean_squared_error(testData, predictions)]], columns=['MSE'])
            resultsDF = pd.concat([resultsDF, error], axis = 1)
            errorStr = mean_squared_error(testData, predictions)
            print('  Test Mean Squared Error: %.3f' % errorStr)
            print()

            chartTitle = str(dataName) + " Training Factor " + str(trainingSet[y])
            resultsDF.to_excel(writer, sheet_name=chartTitle)
            pyplot.title(chartTitle)
            pyplot.xlabel('FT')
            pyplot.ylabel('FN')
            pyplot.plot(testData)
            pyplot.plot(predictionsDF, color='red')
            plot = pyplot
            plotPath = chartTitle + '.png'
            plot.savefig(plotPath)
            
            worksheet = writer.sheets[chartTitle]
            worksheet.insert_image('F1', plotPath)