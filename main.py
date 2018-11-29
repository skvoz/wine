# import numpy as np
import pandas
from clickhouse_driver import Client
import enum
from sklearn.model_selection import train_test_split as train
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale
import numpy as np
from time import mktime

def _is_sunder_fix(name):
    """Returns True if a _sunder_ name, False otherwise.
    Patched: doesn't crash on empty strings
    """
    return (name[:1] == name[-1:] == '_' and
            name[1:2] != '_' and
            name[-2:-1] != '_' and
            len(name) > 2)


enum._is_sunder = _is_sunder_fix
client = Client(
    'ch2-rc.rcrtv.net',
    compression=True
)

sql = '''
        select count(), date from  recreativ.clicks_log_distr group by date limit 10000
    '''
rv = client.execute(sql)

df = pandas.DataFrame(rv, columns = ['hit', 'date'])

X = df.values[::, 0:2]
y = df.values[::, 1:2]
X_train, X_test, y_train, y_test = train(X, y, test_size=0.6)
foo = X_train[::, 0:2]

X_train_draw = scale(X_train[::, 0:2])
X_test_draw = scale(X_test[::, 0:2])
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
y_train_ravel = y_train.ravel()
# print(X_train_draw)
# print(y_train_ravel)

clf.fit(X_train_draw, y_train_ravel)

x_min, x_max = X_train_draw[:, 0].min() - 1, X_train_draw[:, 0].max() + 1
y_min, y_max = X_train_draw[:, 1].min() - 1, X_train_draw[:, 1].max() + 1

h = 0.02

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
            np.arange(y_min, y_max, h))

pred = clf.predict(np.c_[xx.ravel(), yy.ravel()])
pred = pred.reshape(xx.shape)

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

plt.figure()
plt.pcolormesh(xx, yy, pred, cmap=cmap_light)
plt.scatter(X_train_draw[:, 0], X_train_draw[:, 1],
            c=y_train.ravel(), cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.title("Score: %.0f percents" % (clf.score(X_test_draw, y_test) * 100))
plt.show()

print(df)


# path = "wine.csv"
# data = read(path, delimiter=",")
#
# X = data.values[::, 1:14]
# y = data.values[::, 0:1]
#
# from sklearn.cross_validation import train_test_split as train
# X_train, X_test, y_train, y_test = train(X, y, test_size=0.6)
#
# from sklearn.ensemble import RandomForestClassifier
#
# from sklearn.preprocessing import scale
# X_train_draw = scale(X_train[::, 0:2])
# X_test_draw = scale(X_test[::, 0:2])
#
# clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
# clf.fit(X_train_draw, y_train.ravel())
#
# x_min, x_max = X_train_draw[:, 0].min() - 1, X_train_draw[:, 0].max() + 1
# y_min, y_max = X_train_draw[:, 1].min() - 1, X_train_draw[:, 1].max() + 1
#
# h = 0.02
#
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#             np.arange(y_min, y_max, h))
#
# pred = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# pred = pred.reshape(xx.shape)
#
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
#
# cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
# cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
#
# plt.figure()
# plt.pcolormesh(xx, yy, pred, cmap=cmap_light)
# plt.scatter(X_train_draw[:, 0], X_train_draw[:, 1],
#             c=y_train.ravel(), cmap=cmap_bold)
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())
#
# plt.title("Score: %.0f percents" % (clf.score(X_test_draw, y_test) * 100))
# plt.show()
