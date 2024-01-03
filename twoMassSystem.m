% Parameter festlegen
m1 = 1; % Masse 1
m2 = 2; % Masse 2
k1 = 10; % Federkonstante 1
k2 = 5; % Federkonstante 2
b1 = 0.5; % Reibung 1
b2 = 0.2; % Reibung 2

% Zustandsraummodell definieren
A = [0 1 0 0; -(k1/m1) -(b1/m1) (k2/m1) 0; 0 0 0 1; (k2/m2) 0 -(k2/m2) -(b2/m2)];
B = [0; 0; 0; 0];
C = [1 0 -1 0; 0 0 1 0];
D = [0; 0];
sys = ss(A, B, C, D);

% Anfangsbedingungen und Zeitvektor
x0 = [1; 1; 1; 1]; % Anfangszustand
t = 0:0.1:50; % Zeitvektor (0 bis 10 Sekunden)

% Systemantwort berechnen
[y, t_sim, x] = lsim(sys, zeros(size(t)), t, x0);

% Ergebnisse plotten
plot(t_sim, y(:,1), 'r', t_sim, y(:,2), 'b');
xlabel('Zeit');
ylabel('Position');
legend('Masse 1', 'Masse 2');
title('Simulation des Zweimassensystems mit Reibung');
