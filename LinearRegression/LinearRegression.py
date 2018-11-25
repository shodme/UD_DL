# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Read the data
bmi_life_data = pd.read_csv('bmi_and_life_expectancy.csv')
bmi_data = bmi_life_data[['BMI']]
life_data = bmi_life_data[['Life expectancy']]

bmi_data = np.array(bmi_data)
life_data = np.array(life_data)

model = LinearRegression()
model.fit(bmi_data, life_data)

laos_life_exp = model.predict(21.07931)

print(laos_life_exp)

plt.scatter(bmi_data, life_data)
plt.plot(bmi_data, model.predict(bmi_data))
plt.show()
