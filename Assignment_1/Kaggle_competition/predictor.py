import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

pd.options.mode.chained_assignment = None

train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")

train["Sex"] = train["Sex"].fillna("male")
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1
train["Embarked"] = train["Embarked"].fillna("S")

test["Sex"] = test["Sex"].fillna("male")
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1
test["Embarked"] = test["Embarked"].fillna("S")

train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2

train["Pclass"] = train["Pclass"].fillna(0)
train["SibSp"] = train["SibSp"].fillna(train["SibSp"].median())
train["Parch"] = train["Parch"].fillna(train["Parch"].median())
train["Age"] = train["Age"].fillna(train["Age"].median())


test["Pclass"] = test["Pclass"].fillna(0)
test["SibSp"] = test["SibSp"].fillna(train["SibSp"].median())
test["Parch"] = test["Parch"].fillna(train["Parch"].median())
test["Age"] = test["Age"].fillna(train["Age"].median())
test["Fare"] = test["Fare"].fillna(train["Fare"].median())


features_forest = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
target = train["Survived"].values

forest = RandomForestClassifier(max_depth=10, min_samples_split=2, n_estimators=100, random_state=1)

my_forest = forest.fit(features_forest, target)

my_prediction = my_forest.predict(test_features)

PassengerId = np.array(test["PassengerId"]).astype(int)

my_solution = pd.DataFrame(my_prediction, PassengerId, columns=["Survived"])
print(my_solution)
print(my_solution.shape)

my_solution.to_csv("my_solution.csv", index_label= ["PassengerId"])
