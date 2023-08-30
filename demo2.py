
import base64
import io
from flask import Flask, request, jsonify
from flask_cors import CORS

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, mean_squared_error, precision_score, recall_score,r2_score
from sklearn.tree import DecisionTreeClassifier

from collections import Counter
app = Flask(__name__)
CORS(app)


#1.支持向量机
class SVM:
    def __init__(self, kernel='linear'):
        self.kernel = kernel
        self.model = None
    
    def train(self, X, y):
        self.model = SVC(kernel=self.kernel)
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.title('Decision Boundary')
    plt.show()
#2.随机森林
class RandomForest:
    def __init__(self, n_estimators=100, random_state=None):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None
    
    def train(self, X, y):
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state)
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        return accuracy, precision
    
    def confusion_matrix(self, y_true, y_pred):
        return confusion_matrix(y_true, y_pred)
# 3.k均值聚类
class KMeansCluster:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=self.n_clusters)
        
    def train(self, X,y):
        self.model.fit(X,y)
        
    def predict(self, data):
        return self.model.predict(data)
    
    def evaluate(self, data, labels):
        return accuracy_score(labels, self.model.predict(data))
    
    def visualize(self, data):
        labels = self.model.predict(data)
        
        plt.scatter(data[:, 0], data[:, 1], c=labels)
        plt.scatter(self.model.cluster_centers_[:, 0], self.model.cluster_centers_[:, 1], marker='x', color='red')
        plt.show()
# 4.梯度增强
class GradientBoosting:
    def __init__(self):
        self.model = GradientBoostingRegressor()
        
    def train(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        y_pred = self.model.predict(X)
        return mean_squared_error(y, y_pred)
# 5.线性回归 
class MyLinearRegression:
    def __init__(self):
        self.model = None
    
    def train(self, X, y):
        self.model = LinearRegression()
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
#6.决策树
class DecisionTree:
    def __init__(self, criterion='gini', max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.model = None
    
    def train(self, X, y):
        self.model = DecisionTreeClassifier(criterion=self.criterion, max_depth=self.max_depth)
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
#7.k值邻近
class KNN:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.model = None
    
    def train(self, X, y):
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
#8.逻辑回归
class LogisticRegressionModel:
    def __init__(self):
        self.model = None
    
    def train(self, X, y):
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)

#9.朴素贝叶斯
class NaiveBayesModel:
    def __init__(self):
        self.model = None
    
    def train(self, X, y):
        self.model = GaussianNB()
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y_true):
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        return accuracy, f1
#10.降维算法
class DimensionalityReduction:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.model = None
    
    def train(self, X,y):
        self.model = PCA(n_components=self.n_components)
        X_transformed = self.model.fit_transform(X)
        return X_transformed
    
    def transform(self, X):
        if self.model is None:
            raise ValueError("The model is not trained. Please call 'fit_transform()' before 'transform()'.")
        X_transformed = self.model.transform(X)
        return X_transformed

    def predict(self, X):
        return X

class DatasetLoader:
    def load_dataset(self, dataset):
        if dataset == "iris":
            iris = datasets.load_iris()
            return iris
        elif dataset == "diabetes":
            diabetes = datasets.load_diabetes()
            return diabetes
        else:
            print("无效的选择")
            return None
#标准化对象
scaler =   StandardScaler()
class DatasetSplitter:
    def __init__(self, dataset):
        self.dataset = dataset

    def split_dataset(self, split_method, split_ratio=None, n_splits=None):
        if split_method == "RAMDOM":
            X_train, X_test, y_train, y_test = train_test_split(self.dataset.data, self.dataset.target, test_size=split_ratio, random_state=42)
            # 特征标准化
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            return X_train, X_test, y_train, y_test
        elif split_method == "KFOLD":
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            X = self.dataset.data
            y = self.dataset.target
            splits = kf.split(X, y)
            return splits
        elif split_method == "StratifiedKFold":
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            X = self.dataset.data
            y = self.dataset.target
            splits = skf.split(X, y)
            return splits
        else:
            print("无效的划分方式")
            return None

class DataCleaner:
    def __init__(self, dataset):
        self.dataset = dataset

    def check_outliers(self):
        outliers = []
        if isinstance(self.dataset, pd.DataFrame):
            for column in self.dataset.columns:
                q1 = self.dataset[column].quantile(0.25)
                q3 = self.dataset[column].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                column_outliers = self.dataset[(self.dataset[column] < lower_bound) | (self.dataset[column] > upper_bound)].index
                outliers.extend(column_outliers)
        else:
            for column in range(self.dataset.shape[1]):
                column_data = self.dataset[:, column]
                q1 = np.percentile(column_data, 25)
                q3 = np.percentile(column_data, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                column_outliers = np.where((column_data < lower_bound) | (column_data > upper_bound))[0]
                outliers.extend(column_outliers)
        return list(set(outliers))

    def handle_outliers(self):
        outliers = self.check_outliers()
        if isinstance(self.dataset, pd.DataFrame):
            self.dataset = self.dataset.drop(outliers)
        else:
            self.dataset = np.delete(self.dataset, outliers, axis=0)

    def check_missing_values(self):
        if isinstance(self.dataset, pd.DataFrame):
            return self.dataset.isnull().sum().sum()
        else:
            return np.isnan(self.dataset).sum()

    def handle_missing_values(self, strategy="mean"):
        if isinstance(self.dataset, pd.DataFrame):
            self.dataset.fillna(strategy, inplace=True)
        else:
            if strategy == "mean":
                col_means = np.nanmean(self.dataset, axis=0)
                inds = np.where(np.isnan(self.dataset))
                self.dataset[inds] = np.take(col_means, inds[1])
            elif strategy == "median":
                col_medians = np.nanmedian(self.dataset, axis=0)
                inds = np.where(np.isnan(self.dataset))
                self.dataset[inds] = np.take(col_medians, inds[1])
            elif strategy == "most_frequent":
                from scipy import stats
                col_modes = stats.mode(self.dataset, nan_policy='omit')[0].squeeze()
                inds = np.where(np.isnan(self.dataset))
                self.dataset[inds] = np.take(col_modes, inds[1])
            else:
                print("无效的缺失值处理策略")

class ModelSelector:
    def __init__(self, dataset):
        self.dataset = dataset

    def select_model(self, model_name):
        if model_name == "SVM":
            model = SVM()
        elif model_name == "RANDOMFOREST":
            model = RandomForest()
        elif model_name == "KMeansCluster":
            model = KMeansCluster()
        elif model_name == "GradientBoosting":
            model = GradientBoosting()
        elif model_name == "LinearRegression":
            model = MyLinearRegression()
        elif model_name == "DecisionTree":
            model = DecisionTree()
        elif model_name == "KNN":
            model = KNN()
        elif model_name == "LogisticRegressionModel":
            model = LogisticRegressionModel()
        elif model_name == "NaiveBayesModel":
            model = NaiveBayesModel()
        elif model_name == "DimensionalityReduction":
            model = DimensionalityReduction()
        else:
            print("无效的模型选择")
            return None
        return model

class ModelTrainer:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def train_random(self,X_train, y_train):
        #X_train, X_test, y_train, y_test = train_test_split(self.dataset.data, self.dataset.target, test_size=0.2, random_state=42)
        self.model.train(X_train, y_train)

    def train_kfold(self, n_splits):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        X = self.dataset.data
        y = self.dataset.target

        for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            self.model.train(X_train, y_train)
            print("Fold", fold+1)
            #print("训练集准确率:", self.model.score(X_train, y_train))
            #print("测试集准确率:", self.model.score(X_test, y_test))
            return X_test, y_test

    def train_stratified_kfold(self, n_splits):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        X = self.dataset.data
        y = self.dataset.target
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            self.model.train(X_train, y_train)
            print("Fold", fold+1)
            #print("训练集准确率:", self.model.score(X_train, y_train))
            #print("测试集准确率:", self.model.score(X_test, y_test))
            return X_test, y_test
class ModelEvaluator:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.data=datasets

    def evaluate_model(self, model_name):
        y_pred = self.model.predict(self.X_test)
        if model_name == "SVM":
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            mse = mean_squared_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            return {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mse": mse,
                "r2":r2,
            }
        elif model_name == "RANDOMFOREST":
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            mse = mean_squared_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            #classification_report = self.get_classification_report()
            return {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mse": mse,
                "r2":r2
                #"classification_report": classification_report
            }
        else:
            print("无效的模型选择")
            return None

    #def get_classification_report(self):
        #classification_report = classification_report(self.y_test, y_pred)
        #return classification_report
#随机森林评估
def evaluate_random_forest(X, y, split_method,get_rate,get_split):
    if split_method == 'RANDOM':
        # 随机划分数据集为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=get_rate, random_state=42)
        accuracy_scores = []
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy)
        avg_accuracy = np.mean(accuracy_scores)
        return avg_accuracy, y_test, y_pred    
    elif split_method == 'KFOLD':
        # 使用KFold交叉验证划分数据集
        kfold = KFold(n_splits=get_split, shuffle=True, random_state=42)
        accuracy_scores = []
        for train_index, test_index in kfold.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # 创建随机森林模型
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

            # 训练模型
            rf_model.fit(X_train, y_train)

            # 在测试集上进行预测
            y_pred = rf_model.predict(X_test)

            # 计算准确率
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_scores.append(accuracy)

        # 输出平均准确率和测试集上的真实标签与预测标签
        avg_accuracy = np.mean(accuracy_scores)
        return avg_accuracy, y_test, y_pred
    elif split_method == 'STRATIFIEDKFOLD':
        # 使用StratifiedKFold交叉验证划分数据集
        stratified_kfold = StratifiedKFold(n_splits=get_split, shuffle=True, random_state=42)
        accuracy_scores = []
        for train_index, test_index in stratified_kfold.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # 创建随机森林模型
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

            # 训练模型
            rf_model.fit(X_train, y_train)

            # 在测试集上进行预测
            y_pred = rf_model.predict(X_test)

            # 计算准确率
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_scores.append(accuracy)

        # 输出平均准确率和测试集上的真实标签与预测标签
        avg_accuracy = np.mean(accuracy_scores)
        return avg_accuracy, y_test, y_pred
    else:
        return None, None, None
#SVM评估
def evaluate_svm(X, y, split_method,get_rate,get_split):
    if split_method == 'RANDOM':
        # 随机划分数据集为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=get_rate, random_state=42)
        accuracy_scores = []
        svm_model = SVC(kernel='linear', C=1.0, random_state=42)
        svm_model.fit(X_train, y_train)
        y_pred = svm_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy)
        avg_accuracy = np.mean(accuracy_scores)
        return avg_accuracy, y_test, y_pred
    elif split_method == 'KFOLD':
        # 使用KFold交叉验证划分数据集
        kfold = KFold(n_splits=get_split, shuffle=True, random_state=42)
        accuracy_scores = []
        for train_index, test_index in kfold.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # 创建SVM模型
            svm_model = SVC(kernel='linear', C=1.0, random_state=42)

            # 训练模型
            svm_model.fit(X_train, y_train)

            # 在测试集上进行预测
            y_pred = svm_model.predict(X_test)

            # 计算准确率
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_scores.append(accuracy)

        # 输出平均准确率和测试集上的真实标签与预测标签
        avg_accuracy = np.mean(accuracy_scores)
        return avg_accuracy, y_test, y_pred
    elif split_method == 'STRATIFIEDKFOLD':
        # 使用StratifiedKFold交叉验证划分数据集
        stratified_kfold = StratifiedKFold(n_splits=get_split, shuffle=True, random_state=42)
        accuracy_scores = []
        for train_index, test_index in stratified_kfold.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # 创建SVM模型
            svm_model = SVC(kernel='linear', C=1.0, random_state=42)

            # 训练模型
            svm_model.fit(X_train, y_train)

            # 在测试集上进行预测
            y_pred = svm_model.predict(X_test)

            # 计算准确率
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_scores.append(accuracy)

        # 输出平均准确率和测试集上的真实标签与预测标签
        avg_accuracy = np.mean(accuracy_scores)
        return avg_accuracy, y_test, y_pred
    else:
        return None, None, None
#决策树评估
def evaluate_decision_tree(X, y, split_method,get_rate,get_split):
    if split_method == 'RANDOM':
        # 随机划分数据集为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=get_rate, random_state=42)
        accuracy_scores = []
        dt_model = DecisionTreeClassifier(random_state=42)
        dt_model.fit(X_train, y_train)
        y_pred = dt_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy)
        avg_accuracy = np.mean(accuracy_scores)
        return avg_accuracy, y_test, y_pred
    elif split_method == 'KFOLD':
        # 使用KFold交叉验证划分数据集
        kfold = KFold(n_splits=get_split, shuffle=True, random_state=42)
        accuracy_scores = []
        for train_index, test_index in kfold.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # 创建决策树模型
            dt_model = DecisionTreeClassifier(random_state=42)

            # 训练模型
            dt_model.fit(X_train, y_train)

            # 在测试集上进行预测
            y_pred = dt_model.predict(X_test)

            # 计算准确率
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_scores.append(accuracy)

        # 输出平均准确率和测试集上的真实标签与预测标签
        avg_accuracy = np.mean(accuracy_scores)
        return avg_accuracy, y_test, y_pred
    elif split_method == 'STRATIFIEDKFOLD':
        # 使用StratifiedKFold交叉验证划分数据集
        stratified_kfold = StratifiedKFold(n_splits=get_split, shuffle=True, random_state=42)
        accuracy_scores = []
        for train_index, test_index in stratified_kfold.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # 创建决策树模型
            dt_model = DecisionTreeClassifier(random_state=42)

            # 训练模型
            dt_model.fit(X_train, y_train)

            # 在测试集上进行预测
            y_pred = dt_model.predict(X_test)

            # 计算准确率
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_scores.append(accuracy)

        # 输出平均准确率和测试集上的真实标签与预测标签
        avg_accuracy = np.mean(accuracy_scores)
        return avg_accuracy, y_test, y_pred
    else:
        return None, None, None
#knn评估
def evaluate_knn(X, y, split_method,get_rate,get_split):
    if split_method == 'RANDOM':
        # 随机划分数据集为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=get_rate, random_state=42)
        accuracy_scores = []
        knn_model = KNeighborsClassifier(n_neighbors=3)
        knn_model.fit(X_train, y_train)
        y_pred = knn_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy)
        avg_accuracy = np.mean(accuracy_scores)
        return avg_accuracy, y_test, y_pred
    elif split_method == 'KFOLD':
        # 使用KFold交叉验证划分数据集
        kfold = KFold(n_splits=get_split, shuffle=True, random_state=42)
        accuracy_scores = []
        for train_index, test_index in kfold.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # 创建KNN模型
            knn_model = KNeighborsClassifier(n_neighbors=3)

            # 训练模型
            knn_model.fit(X_train, y_train)

            # 在测试集上进行预测
            y_pred = knn_model.predict(X_test)

            # 计算准确率
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_scores.append(accuracy)

        # 输出平均准确率和测试集上的真实标签与预测标签
        avg_accuracy = np.mean(accuracy_scores)
        return avg_accuracy, y_test, y_pred
    elif split_method == 'STRATIFIEDKFOLD':
        # 使用StratifiedKFold交叉验证划分数据集
        stratified_kfold = StratifiedKFold(n_splits=get_split, shuffle=True, random_state=42)
        accuracy_scores = []
        for train_index, test_index in stratified_kfold.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # 创建KNN模型
            knn_model = KNeighborsClassifier(n_neighbors=3)

            # 训练模型
            knn_model.fit(X_train, y_train)

            # 在测试集上进行预测
            y_pred = knn_model.predict(X_test)

            # 计算准确率
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_scores.append(accuracy)

        # 输出平均准确率和测试集上的真实标签与预测标签
        avg_accuracy = np.mean(accuracy_scores)
        return avg_accuracy, y_test, y_pred
    else:
        return None, None, None
#逻辑回归评估
def evaluate_logistic_regression(X, y, split_method,get_rate,get_split):
    if split_method == 'RANDOM':
        # 随机划分数据集为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=get_rate, random_state=42)
        accuracy_scores = []
        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_model.fit(X_train, y_train)
        y_pred = lr_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy)
        avg_accuracy = np.mean(accuracy_scores)
        return avg_accuracy, y_test, y_pred
    elif split_method == 'KFOLD':
        # 使用KFold交叉验证划分数据集
        kfold = KFold(n_splits=get_split, shuffle=True, random_state=42)
        accuracy_scores = []
        for train_index, test_index in kfold.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # 创建逻辑回归模型
            lr_model = LogisticRegression(max_iter=1000, random_state=42)

            # 训练模型
            lr_model.fit(X_train, y_train)

            # 在测试集上进行预测
            y_pred = lr_model.predict(X_test)

            # 计算准确率
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_scores.append(accuracy)

        # 输出平均准确率和测试集上的真实标签与预测标签
        avg_accuracy = np.mean(accuracy_scores)
        return avg_accuracy, y_test, y_pred
    elif split_method == 'STRATIFIEDKFOLD':
        # 使用StratifiedKFold交叉验证划分数据集
        stratified_kfold = StratifiedKFold(n_splits=get_split, shuffle=True, random_state=42)
        accuracy_scores = []
        for train_index, test_index in stratified_kfold.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # 创建逻辑回归模型
            lr_model = LogisticRegression(max_iter=1000, random_state=42)

            # 训练模型
            lr_model.fit(X_train, y_train)

            # 在测试集上进行预测
            y_pred = lr_model.predict(X_test)

            # 计算准确率
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_scores.append(accuracy)

        # 输出平均准确率和测试集上的真实标签与预测标签
        avg_accuracy = np.mean(accuracy_scores)
        return avg_accuracy, y_test, y_pred
    else:
        return None, None, None
#朴素贝叶斯评估
def evaluate_naive_bayes(X, y, split_method,get_rate,get_split):
    if split_method == 'RANDOM':
        # 随机划分数据集为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=get_rate, random_state=42)
        accuracy_scores = []
        nb_model = GaussianNB()
        nb_model.fit(X_train, y_train)
        y_pred = nb_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy)
        avg_accuracy = np.mean(accuracy_scores)
        return avg_accuracy, y_test, y_pred
    elif split_method == 'KFOLD':
        # 使用KFold交叉验证划分数据集
        kfold = KFold(n_splits=get_split, shuffle=True, random_state=42)
        accuracy_scores = []
        for train_index, test_index in kfold.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # 创建朴素贝叶斯模型
            nb_model = GaussianNB()

            # 训练模型
            nb_model.fit(X_train, y_train)

            # 在测试集上进行预测
            y_pred = nb_model.predict(X_test)

            # 计算准确率
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_scores.append(accuracy)

        # 输出平均准确率和测试集上的真实标签与预测标签
        avg_accuracy = np.mean(accuracy_scores)
        return avg_accuracy, y_test, y_pred
    elif split_method == 'STRATIFIEDKFOLD':
        # 使用StratifiedKFold交叉验证划分数据集
        stratified_kfold = StratifiedKFold(n_splits=get_split, shuffle=True, random_state=42)
        accuracy_scores = []
        for train_index, test_index in stratified_kfold.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # 创建朴素贝叶斯模型
            nb_model = GaussianNB()

            # 训练模型
            nb_model.fit(X_train, y_train)

            # 在测试集上进行预测
            y_pred = nb_model.predict(X_test)

            # 计算准确率
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_scores.append(accuracy)

        # 输出平均准确率和测试集上的真实标签与预测标签
        avg_accuracy = np.mean(accuracy_scores)
        return avg_accuracy, y_test, y_pred
    else:
        return None, None, None
#降维评估
def evaluate_pca(X, y, split_method,get_rate,get_split):
    if split_method == 'RANDOM':
        # 随机划分数据集为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=get_rate, random_state=42)
        accuracy_scores = []
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_model.fit(X_train_pca, y_train)
        y_pred = lr_model.predict(X_test_pca)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy)
        avg_accuracy = np.mean(accuracy_scores)
        return avg_accuracy, y_test, y_pred
    elif split_method == 'KFOLD':
        # 使用KFold交叉验证划分数据集
        kfold = KFold(n_splits=get_split, shuffle=True, random_state=42)
        accuracy_scores = []
        for train_index, test_index in kfold.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # 使用PCA进行降维
            pca = PCA(n_components=2)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)

            # 创建逻辑回归模型
            lr_model = LogisticRegression(max_iter=1000, random_state=42)

            # 训练模型
            lr_model.fit(X_train_pca, y_train)

            # 在测试集上进行预测
            y_pred = lr_model.predict(X_test_pca)

            # 计算准确率
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_scores.append(accuracy)

        # 输出平均准确率和测试集上的真实标签与预测标签
        avg_accuracy = np.mean(accuracy_scores)
        return avg_accuracy, y_test, y_pred
    elif split_method == 'STRATIFIEDKFOLD':
        # 使用StratifiedKFold交叉验证划分数据集
        stratified_kfold = StratifiedKFold(n_splits=get_split, shuffle=True, random_state=42)
        accuracy_scores = []
        for train_index, test_index in stratified_kfold.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # 使用PCA进行降维
            pca = PCA(n_components=2)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)

            # 创建逻辑回归模型
            lr_model = LogisticRegression(max_iter=1000, random_state=42)

            # 训练模型
            lr_model.fit(X_train_pca, y_train)

            # 在测试集上进行预测
            y_pred = lr_model.predict(X_test_pca)

            # 计算准确率
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_scores.append(accuracy)

        # 输出平均准确率和测试集上的真实标签与预测标签
        avg_accuracy = np.mean(accuracy_scores)
        return avg_accuracy, y_test, y_pred
    else:
        return None, None, None
#线性回归评估
def evaluate_linear_regression(X, y, split_method,get_rate):
    if split_method == 'RANDOM':
        # 随机划分数据集为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=get_rate, random_state=42)
    else:
        print("无效的划分方式选择。")
        return None, None, None

    # 创建线性回归模型
    lr_model = LinearRegression()

    # 训练模型
    lr_model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = lr_model.predict(X_test)

    # 计算平均绝对误差
    mae = np.mean(np.abs(y_test - y_pred))
    
    return mae, y_test, y_pred
#k均值聚类评估
def evaluate_kmeans(X, split_method,get_rate):
    if split_method == 'RANDOM':
        # 随机划分数据集为训练集和测试集
        X_train, X_test = train_test_split(X, test_size=get_rate, random_state=42)
    else:
        print("无效的划分方式选择。")
        return None, None, None

    # 创建K均值聚类模型
    kmeans_model = KMeans(n_clusters=3, random_state=42)

    # 训练模型
    kmeans_model.fit(X_train)

    # 在测试集上进行预测
    y_pred = kmeans_model.predict(X_test)
    
    return y_pred
#梯度增强评估
def evaluate_gradient_boosting(X, y, split_method,get_rate):
    if split_method == 'RANDOM':
        # 随机划分数据集为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=get_rate, random_state=42)
    else:
        print("无效的划分方式选择。")
        return None, None, None

    # 创建梯度增强模型
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

    # 训练模型
    gb_model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = gb_model.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy, y_test, y_pred
@app.route('/get_data', methods=['POST','GET'])
def get_data():
    #数据接收与处理
    data = request.get_json()
    selected_dataset = data.get('dataset', '')
    rate_get=data.get('split_ratio', '0.3')
    model_selected=data.get('model','')

    Loader=DatasetLoader()
    Loader_data=Loader.load_dataset(selected_dataset)
    X = Loader_data.data
    y = Loader_data.target

    splits_n=data.get('n_splits','')
    method_choose=data.get('split_method','')

    #spilitter=DatasetSplitter(Loader_data)

    #Model_selected=ModelSelector(Loader_data)
    #Model=Model_selected.select_model(model_selected)

    average_accuracy = None
    y_test = None
    y_pred = None

    if model_selected=="SVM":
        average_accuracy, y_test, y_pred = evaluate_svm(X, y, method_choose,rate_get,splits_n)
    elif model_selected=="RANDOMFOREST":
        average_accuracy, y_test, y_pred = evaluate_random_forest(X, y, method_choose,rate_get,splits_n)
    else:
        print("无效的模型选择。")

    if average_accuracy is not None and y_test is not None and y_pred is not None:
        print(f"Average Accuracy ({model_selected}, {method_choose}):", average_accuracy)
        
        # 输出分类报告
        report = classification_report(y_test, y_pred, target_names=Loader_data.target_names)
        print("Classification Report:\n", report)

        # 计算混淆矩阵
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:\n", conf_matrix)

        # 可视化混淆矩阵
        plt.figure(figsize=(8, 6))
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(Loader_data.target_names))
        plt.xticks(tick_marks, Loader_data.target_names, rotation=45)
        plt.yticks(tick_marks, Loader_data.target_names)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
    else:
        print("无效的划分方式选择。")


    return jsonify({"data_name": selected_dataset, 
                    "model_name": model_selected,
                    "split_ratio":rate_get/100,
                    "split_method":method_choose,
                    #"conf_matrix":conf_matrix,
                    "Classification Report":report
                    })

@app.route('/get_data1', methods=['POST','GET'])
def get_data1():
    #数据接收与处理
    data = request.get_json()
    selected_dataset = data.get('dataset', '')
    rate_get=data.get('split_ratio', '0.3')
    model_selected=data.get('model','')

    Loader=DatasetLoader()
    Loader_data=Loader.load_dataset(selected_dataset)
    X = Loader_data.data
    y = Loader_data.target

    splits_n=data.get('n_splits','')
    method_choose=data.get('split_method','')

    average_accuracy = None
    y_test = None
    y_pred = None

    if model_selected == 'DECISIONTREE':
        if method_choose in ['RANDOM', 'KFOLD', 'STRATIFIEDKFOLD']:
            average_accuracy, y_test, y_pred = evaluate_decision_tree(X, y, method_choose,rate_get,splits_n)
        else:
            print("无效的划分方式选择。")
    elif model_selected == 'KNN':
        if method_choose in ['RANDOM', 'KFOLD', 'STRATIFIEDKFOLD']:
            average_accuracy, y_test, y_pred = evaluate_knn(X, y, method_choose,rate_get,splits_n)
        else:
            print("无效的划分方式选择。")
    else:
        print("无效的模型选择。")

    if average_accuracy is not None and y_test is not None and y_pred is not None:
        print(f"Average Accuracy ({model_selected}, {method_choose}):", average_accuracy)
        
        # 输出分类报告
        report = classification_report(y_test, y_pred, target_names=Loader_data.target_names)
        print("Classification Report:\n", report)

        # 计算混淆矩阵
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:\n", conf_matrix)

        # 可视化混淆矩阵
        plt.figure(figsize=(8, 6))
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(Loader_data.target_names))
        plt.xticks(tick_marks, Loader_data.target_names, rotation=45)
        plt.yticks(tick_marks, Loader_data.target_names)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
    else:
        print("无效的划分方式选择。")
    return jsonify({"data_name": selected_dataset, 
                    "model_name": model_selected,
                    "split_ratio":rate_get/100,
                    "split_method":method_choose,
                    #"conf_matrix":conf_matrix,
                    "Classification Report":report
                    })

@app.route('/get_data2', methods=['POST','GET'])
def get_data2():
    #数据接收与处理
    data = request.get_json()
    selected_dataset = data.get('dataset', '')
    rate_get=data.get('split_ratio', '0.3')
    model_selected=data.get('model','')

    Loader=DatasetLoader()
    Loader_data=Loader.load_dataset(selected_dataset)
    X = Loader_data.data
    y = Loader_data.target

    splits_n=data.get('n_splits','')
    method_choose=data.get('split_method','')
    average_accuracy = None
    y_test = None
    y_pred = None

    if model_selected == 'LOGISTIC':
        if method_choose in ['RANDOM', 'KFOLD', 'STRATIFIEDKFOLD']:
            average_accuracy, y_test, y_pred = evaluate_logistic_regression(X, y, method_choose,rate_get,splits_n)
        else:
            print("无效的划分方式选择。")
    elif model_selected == 'NAIVEBAYES':
        if method_choose in ['RANDOM', 'KFOLD', 'STRATIFIEDKFOLD']:
            average_accuracy, y_test, y_pred = evaluate_naive_bayes(X, y, method_choose,rate_get,splits_n)
        else:
            print("无效的划分方式选择。")
    elif model_selected == 'PCA':
        if method_choose in ['RANDOM', 'KFOLD', 'STRATIFIEDKFOLD']:
            average_accuracy, y_test, y_pred = evaluate_pca(X, y, method_choose,rate_get,splits_n)
        else:
            print("无效的划分方式选择。")
    else:
        print("无效的模型选择。")

    if average_accuracy is not None and y_test is not None and y_pred is not None:
        print(f"Average Accuracy ({model_selected}, {method_choose}):", average_accuracy)
        
        # 输出分类报告
        report = classification_report(y_test, y_pred, target_names=Loader_data.target_names)
        print("Classification Report:\n", report)

        # 计算混淆矩阵
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:\n", conf_matrix)

        # 可视化混淆矩阵
        plt.figure(figsize=(8, 6))
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(Loader_data.target_names))
        plt.xticks(tick_marks, Loader_data.target_names, rotation=45)
        plt.yticks(tick_marks, Loader_data.target_names)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
    else:
        print("无效的划分方式选择。")
    return jsonify({"data_name": selected_dataset, 
                    "model_name": model_selected,
                    "split_ratio":rate_get/100,
                    "split_method":method_choose,
                    #"conf_matrix":conf_matrix,
                    #"Classification Report":report
                    })

@app.route('/get_data3', methods=['POST','GET'])
def get_data3():
    #数据接收与处理
    data = request.get_json()
    selected_dataset = data.get('dataset', '')
    rate_get=data.get('split_ratio', '0.3')
    model_selected=data.get('model','')

    Loader=DatasetLoader()
    Loader_data=Loader.load_dataset(selected_dataset)
    X = Loader_data.data
    y = Loader_data.target

    splits_n=data.get('n_splits','')
    method_choose=data.get('split_method','')
    evaluation_result = None
    y_test = None
    y_pred = None

    if model_selected == 'LINEAR':
        evaluation_result, y_test, y_pred = evaluate_linear_regression(X, y, method_choose,rate_get)
    elif model_selected == 'KMEANS':
        y_pred = evaluate_kmeans(X, method_choose,rate_get)
    elif model_selected == 'GRADIENTBOOSTING':
        evaluation_result, y_test, y_pred = evaluate_gradient_boosting(X, y, method_choose,rate_get)
    else:
        print("无效的模型选择。")

    if evaluation_result is not None:
        if model_selected == 'LINEAR':
            print(f"Mean Absolute Error ({model_selected}, {method_choose}):", evaluation_result)
        elif method_choose == 'GRADIENTBOOSTING':
            print(f"Accuracy ({model_selected}, {method_choose}):", evaluation_result)
    else:
        print("无效的划分方式选择。")
    return jsonify({"data_name": selected_dataset, 
                    "model_name": model_selected,
                    "split_ratio":rate_get/100,
                    "split_method":method_choose,
                    #"conf_matrix":conf_matrix,
                    "evaluation_result":evaluation_result
                    })

if __name__ == '__main__':
    app.run()