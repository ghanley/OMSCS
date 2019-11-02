from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

data, labels = load_digits(return_X_y=True)

model = RandomForestRegressor(random_state=1, max_depth=10)
# df=pd.get_dummies(df)
model.fit(data,labels)

features = data.columns
importances = model.feature_importances_
indices = np.argsort(importances)[-9:]  # top 10 features
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
