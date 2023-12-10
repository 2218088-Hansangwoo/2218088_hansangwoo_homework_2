import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data = {
    'Time': range(24),
    'Accidents': [10, 8, 7, 5, 4, 6, 8, 15, 20, 25, 30, 35, 40, 38, 36, 33, 30, 25, 20, 15, 12, 10, 9, 8]
}

df = pd.DataFrame(data)

X = df[['Time']]
y = df['Accidents'] #11.30


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42) #12.01

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

time_range = range(24)
predicted_accidents = model.predict(pd.DataFrame({'Time': time_range}))

plt.plot(time_range, predicted_accidents, label='Predicted Accidents') #12.10
plt.scatter(df['Time'], df['Accidents'], color='red', label='Actual Accidents')
plt.xlabel('Time of Day')
plt.ylabel('Accident Frequency')
plt.title('Predicted vs Actual Accident Frequency by Time of Day')
plt.legend()
plt.show()
