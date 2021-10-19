import numpy as np


class Perceptron(object):
    """パーセプトロンの分類機

    パラメータ
    ----------------
    eta : float
        学習率 (0.0より大きく1.0以下の値)
    n_iter : int
        訓練データの訓練回数
    random_state : int
        重みを初期化するための乱数シード

    属性
    -----------------
    w_ : 1次元配列
        適合後の重み
    errors_ : リスト
        各エポックでのご分類（更新）の数
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1) -> None:
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                # 重みの更新
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                # 重みの更新が０じゃない場合はご分類としてカウント
                errors += int(update != 0.0)
            self.errors_.append(errors)

        return self

    def net_input(self, X: np.ndarray):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """1ステップのあとのクラスのラベルを返す"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
