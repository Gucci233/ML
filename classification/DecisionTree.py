import numpy as np
from collections import Counter


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        y_pred = np.zeros(len(X))
        for i, x in enumerate(X):
            y_pred[i] = self._predict_one(x)
        return y_pred

    def _predict_one(self, x):
        node = self.tree
        while node['left'] is not None and node['right'] is not None:
            if x[node['feature']] <= node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node['class']

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        # 终止条件1：所有样本属于同一类别，无需再分
        if n_classes == 1:
            return {'class': y[0]}
        # 终止条件2：节点样本数小于等于min_samples_leaf，无需再分
        if n_samples <= self.min_samples_leaf:
            return {'class': np.bincount(y).argmax()}
        # 终止条件3：达到最大深度，无需再分
        if self.max_depth is not None and depth == self.max_depth:
            return {'class': np.bincount(y).argmax()}
        # 找到最优划分特征和划分点
        best_feature, best_threshold = self._find_best_split(X, y)
        # 终止条件4：无法再进行有效的划分，无需再分
        if best_feature is None or best_threshold is None:
            return {'class': np.bincount(y).argmax()}
        # 创建节点
        left_idxs = X[:, best_feature] <= best_threshold
        right_idxs = X[:, best_feature] > best_threshold
        node = {'feature': best_feature, 'threshold': best_threshold}
        # 递归构建左右子树
        node['left'] = self._build_tree(X[left_idxs], y[left_idxs], depth+1)
        node['right'] = self._build_tree(X[right_idxs], y[right_idxs], depth+1)
        print(11)
        return node

    def _find_best_split(self, X, y):
        n_samples, n_features = X.shape
        # 初始化最优划分
        best_feature = None
        best_threshold = None
        max_gain = -np.inf
        # 计算基尼系数
        Gini_D = self._gini(y)
        # 在每个特征上进行划分，找到最优划分
        for feature in range(n_features):
            feature_values = np.unique(X[:, feature])
            thresholds = (feature_values[:-1] + feature_values[1:]) / 2.0
            for threshold in thresholds:
                left_idxs = X[:, feature] <= threshold
                right_idxs = X[:, feature] > threshold
                # 终止条件5：分裂后左右子树样本数小于等于min_samples_split，无效分裂
                if np.sum(left_idxs) <= self.min_samples_split or np.sum(right_idxs) <= self.min_samples_split:
                    continue
                # 计算分裂后的基尼系数
                Gini_left = self._gini(y[left_idxs])
                Gini_right = self._gini(y[right_idxs])
                Gini_split = (np.sum(left_idxs) / n_samples) * Gini_left + (np.sum(right_idxs) / n_samples) * Gini_right
                # 计算信息增益
                gain = Gini_D - Gini_split
                # 更新最优划分
                if gain > max_gain:
                    max_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    def _gini(self, y):
        n_samples = len(y)
        gini = 1.0
        counts = Counter(y)
        for label, count in counts.items():
            prob = count / n_samples
            gini -= prob ** 2
        return gini

    def _prune(self, node, X, y):
        if node['left'] is None and node['right'] is None:
            return
        left_idxs = X[:, node['feature']] <= node['threshold']
        right_idxs = X[:, node['feature']] > node['threshold']
        if node['left'] is not None:
            self._prune(node['left'], X[left_idxs], y[left_idxs])
        if node['right'] is not None:
            self._prune(node['right'], X[right_idxs], y[right_idxs])
        # 计算当前节点的分类错误率
        y_pred = np.zeros(len(X))
        for i, x in enumerate(X):
            y_pred[i] = self._predict_one(x)
        error_rate = np.sum(y_pred != y) / len(y)
        # 计算剪枝后的分类错误率
        counts = Counter(y)
        max_count = max(counts.values())
        majority_error_rate = (len(y) - max_count) / len(y)
        # 判断是否进行剪枝
        if error_rate >= majority_error_rate:
            node['left'] = None
            node['right'] = None
            node['class'] = np.bincount(y).argmax()

    def prune(self, X, y):
        self._prune(self.tree, X, y)
        
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 加载数据集
iris = load_iris()
X, y = iris.data, iris.target

# 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建 CART 决策树实例
clf = DecisionTree(max_depth=3, min_samples_split=2, min_samples_leaf=1)

# 训练模型
clf.fit(X_train, y_train)

# 对测试集进行预测
y_pred = clf.predict(X_test)

# 计算准确率
accuracy = np.sum(y_pred == y_test) / len(y_test)
print('Accuracy:', accuracy)


    
