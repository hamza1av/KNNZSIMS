import numpy as np
from scipy.signal import StateSpace, lsim

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

# y_out enthält die simulierten Ausgangsgrößen
# t_out enthält die Zeitpunkte der Simulation
# x_out enthält die Zustände des Systems während der Simulation
# Speichern der simulierten Ausgangsgrößen in einer .npy-Datei
np.save('simulated_outputs.npy', y_out)
# Speichern der Zeitpunkte der Simulation in einer .npy-Datei
np.save('simulated_inputs', u)

u = np.random.rand(len(t))*25
t_out, y_out, x_out = lsim(sys, U=u, T=t, X0=x0)

np.save('test_outputs.npy', y_out)
# Speichern der Zeitpunkte der Simulation in einer .npy-Datei
np.save('test_input.npy', u)
