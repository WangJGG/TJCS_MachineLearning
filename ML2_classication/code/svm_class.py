import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x1, x2, p=3):
    return (1 + np.dot(x1, x2)) ** p

# rbf
def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x-y) ** 2 / (2 * sigma ** 2))

class SVM(object):
    def __init__(self, kernel=linear_kernel, C=None):
        self.kernel = kernel

        # C=0: 硬间隔； 如果C不为0：软间隔
        self.C = C
        if self.C is not None: self.C = float(self.C)

    def train(self, X, y):
        n_samples, n_features = X.shape

        # 计算核矩阵 K
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel(X[i], X[j])

        # 构建 cvxopt 矩阵
        P = cvxopt.matrix(np.outer(y, y) * K, tc='d')
        q = cvxopt.matrix(np.ones(n_samples) * -1, tc='d')
        A = cvxopt.matrix(y, (1, n_samples), tc='d')
        b = cvxopt.matrix(0.0, tc='d')

        # 设置 G 和 h 矩阵
        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1), tc='d')
            h = cvxopt.matrix(np.zeros(n_samples), tc='d')
        else:
            G_top = np.diag(np.ones(n_samples) * -1)
            G_bottom = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((G_top, G_bottom)), tc='d')

            h_top = np.zeros(n_samples)
            h_bottom = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((h_top, h_bottom)), tc='d')

        # 求解二次规划问题
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        a = np.ravel(solution['x'])

        # 识别支持向量并限制数量不超过 448
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]


        self.a = a[ind]
        self.sv = X[ind]
        self.sv_y = y[ind]
        print(f'{len(self.a)} support vectors out of {n_samples} points')

        # 计算 b
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n], ind])
        self.b /= len(self.a)

        # 计算 w（仅在线性核下）
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None



    def project(self, X):
        if self.w is not None:
            # 线性核
            return np.dot(X, self.w) + self.b
        else:
            # 非线性核
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s

            return y_predict + self.b


    def predict(self, X):
        return np.sign(self.project(X))

if __name__ == '__main__':
    import pylab as pl

    # 线性可分
    def gen_lin_separable_data():
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[0.8, 0.6], [0.6, 0.8]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y1 = np.ones(len(X1))
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    # 线性不可分-非线性
    def gen_non_lin_separable_data():
        mean1 = [-1, 2]
        mean2 = [1, -1]
        mean3 = [4, -4]
        mean4 = [-4, 4]
        cov = [[1.0, 0.8], [0.8, 1.0]]
        X1= np.random.multivariate_normal(mean1, cov, 50)
        X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 50)))
        X2 = np.random.multivariate_normal(mean2, cov, 50)
        X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 50)))
        y1 = np.ones(len(X1))
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    # 线性不可分：有干扰项
    def gen_lin_separable_overlap_data():
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[1.5, 1.0], [1.0, 1.5]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y1 = np.ones(len(X1))
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def split_train(X1, y1, X2, y2):
        X1_train = X1[:90]
        X2_train = X2[:90]
        y1_train = y1[:90]
        y2_train = y2[:90]
        X_train = np.vstack((X1_train, X2_train))
        y_train = np.hstack((y1_train, y2_train))
        return X_train, y_train

    def split_test(X1, y1, X2, y2):
        X1_test = X1[:90]
        X2_test = X2[:90]
        y1_test = y1[:90]
        y2_test = y2[:90]
        X_test = np.vstack((X1_test, X2_test))
        y_test = np.hstack((y1_test, y2_test))
        return X_test, y_test

    def plot_margin(X1_train, X2_train, clf):
        def f(x, w, b, c=0):
            return (-w[0] * x -b + c) / w[1]

        pl.plot(X1_train[:, 0], X1_train[:, 1] , 'ro')
        pl.plot(X2_train[:, 0], X2_train[:, 1] , 'bo')
        pl.scatter(clf.sv[:, 0], clf.sv[:, 1], s=100, c='g')

        # w.x + b = 0
        a0 = -4;a1 = f(a0, clf.w, clf.b)
        b0 = 4 ;b1 = f(b0, clf.w, clf.b)
        pl.plot([a0, b0], [a1, b1], 'k')

        # w.x + b = 1
        a0 = -4; a1 = f(a0, clf.w, 1)
        b0 = 4; b1 = f(b0, clf.w, 1)
        pl.plot([a0, b0], [a1, b1], 'k--')

        # w.x + b = -1
        a0 = -4; a1 = f(a0, clf.w, -1)
        b0 = 4 ;b1 = f(b0, clf.w, -1)
        pl.plot([a0, b0], [a1, b1], 'k--')

        pl.axis('tight')
        pl.show()

    def plot_contor(X1_train, X2_train, clf):
        pl.plot(X1_train[:, 0], X1_train[:, 1], 'ro')
        pl.plot(X2_train[:, 0], X2_train[:, 1], 'bo')
        pl.scatter(clf.sv[:, 0], clf.sv[:, 1], s=100, c='g')

        X1, X2 = np.meshgrid(np.linspace(-6, 6, 50), np.linspace(-6, 6, 50))
        X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
        z = clf.project(X).reshape(X1.shape)
        pl.contour(X1, X2, z, [0.0], colors='k', linewidths=1, origin='lower')
        pl.contour(X1, X2, z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
        pl.contour(X1, X2, z - 1, [0.0], colors='grey', linewidths=1, origin='lower')

        pl.axis('tight')
        pl.show()

    def test_linear():
        X1, y1, X2, y2 = gen_lin_separable_data()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        clf =SVM()
        clf.train(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print('%d out of %d predictions correct.' % (correct, len(y_predict)))
        plot_margin(X_train[y_train==1], X_train[y_train==-1], clf)


    def test_non_linear():
        X1, y1, X2, y2 = gen_non_lin_separable_data()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        clf = SVM(polynomial_kernel)
        clf.train(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print('%d out of %d predictions correct.' % (correct, len(y_predict)))
        plot_contor(X_train[y_train == 1], X_train[y_train == -1], clf)

    def test_soft():
        X1, y1, X2, y2 = gen_lin_separable_overlap_data()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        clf = SVM(C=1000.1)
        clf.train(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print('%d out of %d predictions correct.' % (correct, len(y_predict)))
        plot_contor(X_train[y_train == 1], X_train[y_train == -1], clf)

    # test_linear()
    # test_soft()
    test_non_linear()