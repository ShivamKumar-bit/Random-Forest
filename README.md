First we will talk about **Ensemble Techniques** and **Bagging**.
#### Ensemble Techniques
It basically means combining different models, train them from your data set and use it for your predictions.
#### Bagging
One of the ensemble techniques is **Bagging** a.k.a Bootstrap aggregation. Let's say if you have a data set (D) and you have some base learners/ models.
Now  we will divide our data set into say (D1,D2...) using row sampling with replacement technique. And feed these datasets to the models.
###### Note-
Your datasets may have some same data but they won't be equal.
When we give a new test data , your models will predict (if it's a classification problem) 0 or 1 and then we take the  majority answer.
<br>
![Image](https://gaussian37.github.io/assets/img/ml/concept/bagging/bagging.png)
<br>
# Random Forest
Think it as a collection of decision trees. In simple words, apply same concept of bagging . Here the models used will be **decision tree** .
**Why so many Decision Trees?**

Decision Tree has 2 properties:
- Low Bias :low bias means your training error is very less.
- High Variance : when we have new test data our decision tree is prone to error/ overfitting.
Now comes the answer to your **why** Random Forest?
<br>

![Image](http://www.quickmeme.com/img/de/de946157581984180c7402c7b4bf85b92589c505ee10270edf31c22af57a4a0f.jpg)
<br>
In Random Forest we use multiple Decision Tree and each of the Decision tree might have high variance.
But when we combine all the Decision Tree your high variance gets converted to low variance. **How?**
<br>
See each Decision Tree becomes expert with respect to the data sets on which they were trained before.
And we are also not dependent on one Decision Tree instead we take majority  vote from all so High Variance gets converted into Low Variance.
<br>
### Advantage 
Even small changes in your original dataset doesn't make a big difference ,that's the benefit of dividing dataset and taking majority vote.

## Let's code-
```ruby
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.datasets import load_iris ## we will be using iris dataset
```
Some standard imports
```ruby
iris = load_iris()
iris.keys()
```
To get dict_keys
```ruby
df = pd.DataFrame(iris.data,columns=iris.feature_names)
df.head()
```
Created a data frames
```ruby
df['target']=iris.target
df.head()
```
Appending target column i.e 0,1,2
```ruby
sns.pairplot(df,hue='target',palette='Set1')
```

![Image](https://github.com/ShivamKumar-bit/Random-Forest/blob/master/download%20(1).png?raw=true)

```ruby
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop(['target'],axis='columns'),iris.target,test_size=0.1)
```
Importing train_test_split from sklearn
```ruby
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=1) ## you can change these parameters to see change in accuracy
model.fit(X_train, y_train)
```
```ruby
pred = model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,pred))
```
Prints a classification report
```ruby
gmodel = RandomForestClassifier(n_estimators=10)
gmodel.fit(X_train, y_train)
gpred = gmodel.predict(X_test)
print(classification_report(y_test,gpred))
```
Changing parameter of the Random Forest classifier,just to increase efficiency

```ruby
cm= confusion_matrix(y_test,gpred)
plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True)
plt.xlabel("predicted")
plt.ylabel("truth")
```
![Image](https://github.com/ShivamKumar-bit/Random-Forest/blob/master/download%20(2).png?raw=true)
