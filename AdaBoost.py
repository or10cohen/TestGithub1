import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, metrics, tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate, train_test_split

# dataset = datasets.load_wine()
# print(type(data_set))
# print(data_set.keys())
# X = data_set.data
# y = data_set.target
# target2 = np.where(y == 2)
# print(target2)
# X, y = np.delete(X, target2, axis=0), np.delete(y, target2, axis=0)
# y[y == 0] = -1
# print('size y:' ,y.size)
#
# ##---------------------Weak learner Decision Stump-------------------------------------------------------
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# # print(X_train, '\n', y_train, '\n', X_test, '\n', y_test)
# Tree_model = DecisionTreeClassifier(criterion="entropy", max_depth=1)
# model = Tree_model.fit(X_train, y_train)
# # predictions = np.mean(cross_validate(Tree_model,X,Y,cv=100)['test_score'])
# # print('The accuracy is: ',predictions*100,'%')
# y_pred = model.predict(X_test)
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# ####--------------------------------------------------------------##
# fig = plt.figure(figsize=(25,20))
# _ = tree.plot_tree(model,
#                    feature_names=data_set.feature_names,
#                    class_names=data_set.target_names,
#                    filled=True)
# fig.savefig("decistion_tree.png")
# ####--------------------------------------------------------------##

class Boosting:
    def __init__(self, dataset, T, test_dataset):
        self.dataset = dataset
        self.T = T
        self.test_dataset = test_dataset
        self.alphas = None
        self.models = None
        self.accuracy = []
        self.predictions = None
    def fit(self):
        # Set the descriptive features and the target feature
        X = self.dataset.drop(['target'], axis=1)
        y = self.dataset['target'].where(self.dataset['target'] == 1, -1)
        # Initialize the weights of each sample with wi = 1/N and create a dataframe in which the evaluation is computed
        Evaluation = pd.DataFrame(y.copy())
        Evaluation['weights'] = 1 / len(self.dataset)  # Set the initial weights w = 1/N
        # Run the boosting algorithm by creating T "weighted models"
        alphas = []
        models = []
        for t in range(self.T):
            # Train the Decision Stump(s)
            Tree_model = DecisionTreeClassifier(criterion="entropy",
                                                max_depth=1)  # Mind the deth one --> Decision Stump
            # We know that we must train our decision stumps on weighted datasets where the weights depend on the results of
            # the previous decision stumps. To accomplish that, we use the 'weights' column of the above created
            # 'evaluation dataframe' together with the sample_weight parameter of the fit method.
            # The documentation for the sample_weights parameter sais: "[...] If None, then samples are equally weighted."
            # Consequently, if NOT None, then the samples are NOT equally weighted and therewith we create a WEIGHTED dataset
            # which is exactly what we want to have.
            model = Tree_model.fit(X, y, sample_weight=np.array(Evaluation['weights']))
            # Append the single weak classifiers to a list which is later on used to make the
            # weighted decision
            models.append(model)
            predictions = model.predict(X)
            score = model.score(X, y)
            # Add values to the Evaluation DataFrame
            Evaluation['predictions'] = predictions
            Evaluation['evaluation'] = np.where(Evaluation['predictions'] == Evaluation['target'], 1, 0)
            Evaluation['misclassified'] = np.where(Evaluation['predictions'] != Evaluation['target'], 1, 0)
            # Calculate the misclassification rate and accuracy
            accuracy = sum(Evaluation['evaluation']) / len(Evaluation['evaluation'])
            misclassification = sum(Evaluation['misclassified']) / len(Evaluation['misclassified'])
            # Caclulate the error
            err = np.sum(Evaluation['weights'] * Evaluation['misclassified']) / np.sum(Evaluation['weights'])
            # Calculate the alpha values
            alpha = np.log((1 - err) / err)
            alphas.append(alpha)
            # Update the weights wi --> These updated weights are used in the sample_weight parameter
            # for the training of the next decision stump.
            Evaluation['weights'] *= np.exp(alpha * Evaluation['misclassified'])
            # print('The Accuracy of the {0}. model is : '.format(t+1),accuracy*100,'%')
            # print('The missclassification rate is: ',misclassification*100,'%')

        self.alphas = alphas
        self.models = models

    def predict(self):
        X_test = self.test_dataset.drop(['target'], axis=1).reindex(range(len(self.test_dataset)))
        Y_test = self.test_dataset['target'].reindex(range(len(self.test_dataset))).where(self.dataset['target'] == 1,-1)
        # With each model in the self.model list, make a prediction

        accuracy = []
        predictions = []

        for alpha, model in zip(self.alphas, self.models):
            prediction = alpha * model.predict(
                X_test)  # We use the predict method for the single decisiontreeclassifier models in the list
            predictions.append(prediction)
            self.accuracy.append(
                np.sum(np.sign(np.sum(np.array(predictions), axis=0)) == Y_test.values) / len(predictions[0]))
        self.predictions = np.sign(np.sum(np.array(predictions), axis=0))


dataset = datasets.load_wine()
df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
df["target"] = dataset.target

fig = plt.figure(figsize=(10, 10))
ax0 = fig.add_subplot(111)

number_of_base_learners = 5

for i in range(number_of_base_learners):
    model = Boosting(df, i, df)
    model.fit()
    model.predict()

ax0.plot(range(len(model.accuracy)), model.accuracy, '-b')
ax0.set_xlabel('# models used for Boosting ')
ax0.set_ylabel('accuracy')
print('With a number of ', number_of_base_learners, 'base models we receive an accuracy of ', model.accuracy[-1] * 100,
      '%')

plt.savefig('plot.png')












