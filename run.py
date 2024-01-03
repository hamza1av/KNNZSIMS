import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from scipy.signal import StateSpace, lsim
import matplotlib.pyplot as plt

# Parameter festlegen
m1 = 1  # Masse 1
m2 = 2  # Masse 2
k1 = 10  # Federkonstante 1
k2 = 5  # Federkonstante 2
b1 = 0.5  # Reibung 1
b2 = 0.2  # Reibung 2

# Zustandsraummodell definieren
A = np.array([
    [0, 1, 0, 0],
    [-(k1/m1), -(b1/m1), k2/m1, 0],
    [0, 0, 0, 1],
    [k2/m2, 0, -(k2/m2), -(b2/m2)]
])

B = np.array([
    [0],
    [1],
    [0],
    [0]
])

C = np.array([
    [1, 0, -1, 0],
    [0, 0, 1, 0]
])

D = np.array([
    [0],
    [0]
])

# System erstellen
sys = StateSpace(A, B, C, D)

# Zeitpunkte für die Simulation
t = np.linspace(0, 10, 1000)  # von 0 bis 10 Sekunden, 1000 Schritte
# Anregungssignal (hier: keine Anregung)
u = np.random.rand(len(t))*20

# Anfangszustand (hier: Nullzustand)
x0 = np.zeros((4,))

# System simulieren
t_out, y_out, x_out = lsim(sys, U=u, T=t, X0=x0)

##################################################


# Load the model
loaded_model = tf.keras.models.load_model('twoMassesNeuralNetwork.h5')

# Assuming you have new data for prediction
# Replace this with your actual data
# Use the predictions as needed
# Assuming true_outputs are the actual output data
# Make predictions using the loaded model
predictions = loaded_model.predict(u)
 # Replace with your desired usage of predictions
plt.plot(t_out, y_out[:, 0], label='Output 1',color='red')
plt.plot(t_out, y_out[:, 1], label='Output 2',color='blue')
plt.plot(t_out, predictions[:, 1], label='Prediction 1',color='blue',linestyle='dashed')
plt.plot(t_out, predictions[:, 1], label='Prediction 2',color='blue',linestyle='dashed')
plt.xlabel('Time')
plt.ylabel('Outputs')
plt.title('Simulated Outputs over Time')
plt.legend()
plt.grid(True)
plt.show()
# Calculate evaluation metrics
mse = mean_squared_error(y_out, predictions)
mae = mean_absolute_error(y_out, predictions)
r2 = r2_score(y_out, predictions)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R²-Score:", r2)
