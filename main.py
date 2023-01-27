from sklearn.linear_model import LinearRegression
from random import randint

TRAIN_SET_LIMIT = 1000
TRAIN_SET_COUNT = 100

TRAIN_INPUT = list()
TRAIN_OUTPUT = list()
for i in range(TRAIN_SET_COUNT):
    a = randint(0, TRAIN_SET_LIMIT)
    b = randint(0, TRAIN_SET_LIMIT)
    c = randint(0, TRAIN_SET_LIMIT)
    op = (a + b + c) / 3
    TRAIN_INPUT.append([a, b, c])
    TRAIN_OUTPUT.append(op)


predictor = LinearRegression(n_jobs=-1)
predictor.fit(X=TRAIN_INPUT, y=TRAIN_OUTPUT)

X_TEST = [[8, 3, 7]]
out = predictor.predict(X=X_TEST)

print('Out: {}'.format(out))
