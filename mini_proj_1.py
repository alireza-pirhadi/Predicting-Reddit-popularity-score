import numpy as np
import matplotlib as plot
import proj1_data_loading as dataSet

training_set = dataSet.data[0:10000]
validation_set = dataSet.data[10000:11000]
test_set = dataSet.data[11000:12000]
x = np.zeros(shape=[10000, 164])
y = np.zeros(shape=[10000, 1])
most_repeated_words = []
num_repeat = []
# for j in range(10000):
#     print(j)
#     text = str(training_set[j]["text"])
#     text = text.lower()
#     text_words = text.split()
#     for i in range(len(text_words)):
#         if text_words[i] in most_repeated_words:
#             index = most_repeated_words.index(text_words[i])
#             num_repeat[index] += 1
#             while index > 0 and num_repeat[index] > num_repeat[index - 1]:
#                 tmp = num_repeat[index]
#                 num_repeat[index] = num_repeat[index - 1]
#                 num_repeat[index - 1] = tmp
#                 most_repeated_words.remove(text_words[i])
#                 index -= 1
#                 most_repeated_words.insert(index, text_words[i])
#         else:
#             most_repeated_words.append(text_words[i])
#             num_repeat.append(1)
# for i in range(1000):
#     print(str(most_repeated_words[i]) , num_repeat[i] , file = open("most repeated words.txt","a+"))
file = open("most repeated words.txt", "r")
for i in range(160):
    most_repeated_words.append(file.readline().split()[0])
for i in range(10000):
    y[i] = training_set[i]["popularity_score"]
    x[i, 0] = 1
    x[i, 1] = training_set[i]["children"]
    # x[i, 2] = training_set[i]["children"] * training_set[i]["children"]
    x[i, 2] = training_set[i]["controversiality"]
    x[i, 3] = training_set[i]["is_root"]
    text = training_set[i]["text"]
    for j in range(160):
        x[i, 4 + j] = text.count(most_repeated_words[j])

w = np.matmul(np.linalg.inv(np.matmul(x.transpose(), x)), np.matmul(x.transpose(), y))
predicted_score_validationSet = np.zeros(shape=[1000, 1])
predicted_score_testSet = np.zeros(shape=[1000, 1])
predicted_score_trainingSet = np.zeros(shape=[10000, 1])
validation_matrix = np.zeros(shape=[1000, 164])
test_matrix = np.zeros(shape=[1000, 164])
for i in range(1000):
    validation_matrix[i, 0] = 1
    validation_matrix[i, 1] = validation_set[i]["children"]
    # validation_matrix[i, 2] = validation_set[i]["children"] * training_set[i]["children"]
    validation_matrix[i, 2] = validation_set[i]["controversiality"]
    validation_matrix[i, 3] = validation_set[i]["is_root"]
    text = validation_set[i]["text"]
    for j in range(160):
        validation_matrix[i, 4 + j] = text.count(most_repeated_words[j])

for i in range(1000):
    test_matrix[i, 0] = 1
    test_matrix[i, 1] = test_set[i]["children"]
    # validation_matrix[i, 2] = validation_set[i]["children"] * training_set[i]["children"]
    test_matrix[i, 2] = test_set[i]["controversiality"]
    test_matrix[i, 3] = test_set[i]["is_root"]
    text = test_set[i]["text"]
    for j in range(160):
        test_matrix[i, 4 + j] = text.count(most_repeated_words[j])

predicted_score_validationSet = np.matmul(validation_matrix, w)
predicted_score_testSet = np.matmul(test_matrix, w)
predicted_score_trainingSet = np.matmul(x, w)
MSE_validationSet = 0
MSE_testSet = 0
MSE_trainingSet = 0
for i in range(10000):
    MSE_trainingSet += (y[i, 0] - predicted_score_trainingSet[i, 0]) * (y[i, 0] - predicted_score_trainingSet[i, 0])
for i in range(1000):
    y[i] = validation_set[i]["popularity_score"]
    MSE_validationSet += (y[i, 0] - predicted_score_validationSet[i, 0]) * (y[i, 0] - predicted_score_validationSet[i, 0])
for i in range(1000):
    y[i] = test_set[i]["popularity_score"]
    MSE_testSet += (y[i, 0] - predicted_score_testSet[i, 0]) * (y[i, 0] - predicted_score_testSet[i, 0])
print("MSE in trainingSet : " + str(MSE_trainingSet/10000))
print("MSE in validationSet : " + str(MSE_validationSet/1000))
print("MSE in testSet : " + str(MSE_testSet/1000))