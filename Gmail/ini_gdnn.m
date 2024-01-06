%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%                      Datei ini.gdnn.m                           %%%%%
%%%%%             Initialisierung f�r ein GDNN-Netzwerk               %%%%%
%%%%%                      (c) 16.11.2021                             %%%%%
%%%%%                Christian Endisch, TU M�nchen                    %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
clc;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                    Allgemeine Initialisierungen                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Sample Time f�r Simulation
T_sim = 0.01;

%Sample Time f�r neuronales Netz
T = 0.01;

%Simulationsdauer in Sekunden
train_length = 5000;
m = 0.1; 
k= 5 ;
b = 1.2  ;
f = 1.1 ; 
g = 9.81;  
r0 = 1 ; 
g1 = 1 ;
g2 = 1 ;
min_ampl = -1;  %-1;
max_ampl =  1;  % 1;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        Parameteroptimierung                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Fenstergr��e (in Abtastschritten)
Q = 500;

%Levenberg-Marquardt-Parameter
mue = 3;
theta = 10;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           APRBS-Anregung                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Minimale und maximale Amplitude der Trainingsdaten
min_ampl = -10;  %-1;
max_ampl =  10;  % 1;

%Minimaler und maximaler Zeitbereich des APRBS-Signals in Sekunden
t_min = 0.01;  %T;
t_max = 0.25;  %25*T;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           Netzaufbau                                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Zahl der Eing�nge
numofinputs = 1;

%Zahl der Schichten
numoflayers = 3;

%Ausgabe der Gewichte (und Verwaltungsmatrizen) nach der Simulation
print_weights = 1;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             ACHTUNG! hier keine �nderungen vornehmen!                   %
DI = zeros(numoflayers,numofinputs);                                      %
DL = zeros(numoflayers);                           %%%%%%%%%%%%%%         %
idel = zeros(numoflayers,numofinputs,1);            %%        %%          %
ldel = zeros(numoflayers,numoflayers,1);             %%      %%           %
DIS = zeros(numoflayers,numofinputs);                 %%    %%            %
DLS = zeros(numoflayers);                              %%  %%             %
idels = zeros(numoflayers,numofinputs,1);               %%%%              %
ldels = zeros(numoflayers,numoflayers,1);                %%               %
bs = zeros(1,numoflayers);                                                %
DIB = zeros(numoflayers,numofinputs);                   %%%%              %
DLB = zeros(numoflayers);                               %%%%              %
idelb = zeros(numoflayers,numofinputs,1);                                 %
ldelb = zeros(numoflayers,numoflayers,1);                                 %
bb = zeros(1,numoflayers);                                                %
U_target = zeros(1,numoflayers);                                          %
U_out = zeros(1,numoflayers);                                             %
tp = tp_input(train_length/T,[t_min,t_max],[min_ampl,max_ampl],T);        %
%                              Ende                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           Netzaufbau                                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Delay Lines an den Eing�ngen
%[DI, idel] = setDI(toLayer, fromLayer, delayVector, DI, idel)
[DI,idel] = setDI(1,1,[1:3],DI,idel);

%Delay Lines zwischen den Layern
%[DL, ldel] = setDL(toLayer, fromLayer, delayVector, DL, ldel)
[DL,ldel] = setDL(1,1,[1:3],DL,ldel);
[DL,ldel] = setDL(1,2,[1:3],DL,ldel);
[DL,ldel] = setDL(1,3,[1:3],DL,ldel);
[DL,ldel] = setDL(2,1,[0],DL,ldel);         %Normale Forward connection
[DL,ldel] = setDL(2,2,[1:3],DL,ldel);
[DL,ldel] = setDL(2,3,[1:3],DL,ldel);
[DL,ldel] = setDL(3,2,[0],DL,ldel);         %Normale Forward connection
[DL,ldel] = setDL(3,3,[1:3],DL,ldel);

%Dimensionen der Eing�nge
R(1) = 1;

%Zahl der Neuronen pro Schicht
S(1) = 3;
S(2) = 3;
S(3) = 1;

%layer outputs compared to target
U_target(3) = 1;

%layer outputs lead out of S-Function
U_out(3) = 1;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                             Pruning                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%minimal number of weights when pruning
min_weights = 0;

%number of time steps between pruning actions
prune_interval = 100;

%max error (* old error) when taking a pruning step
prune_maxerr_OBS = 2;

%max error (* old error) when taking a pruning step
prune_maxerr_exact = 1; %nur wirsam bei Pruning 3




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        Short Term Memory                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Eing�nge
%[DIS, idels] = setDIS(toLayer, fromLayer, delayVector, DIS, idels)
[DIS,idels] = setDIS(1,1,[],DIS,idels);

%Layer
%[DLS, ldels] = setDLS(toLayer, fromLayer, delayVector, DLS, ldels)
[DLS,ldels] = setDLS(1,1,[],DLS,ldels);
[DLS,ldels] = setDLS(1,2,[],DLS,ldels);
[DLS,ldels] = setDLS(1,3,[],DLS,ldels);
[DLS,ldels] = setDLS(2,1,[],DLS,ldels);         %Normale Forward connection
[DLS,ldels] = setDLS(2,2,[],DLS,ldels);
[DLS,ldels] = setDLS(2,3,[],DLS,ldels);
[DLS,ldels] = setDLS(3,2,[],DLS,ldels);        %Normale Forward connection
[DLS,ldels] = setDLS(3,3,[],DLS,ldels);

%Biases
bs(1) = 0;  %0 oder 1
bs(2) = 0;  %0 oder 1
bs(3) = 0;  %0 oder 1




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                          Block Pruning                                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Eing�nge
%[DIB, idelb] = setDIB(toLayer, fromLayer, delayVector, DIB, idelb)
[DIB,idelb] = setDIB(1,1,[],DIB,idelb);

%Layer
%[DLB, ldelb] = setDLB(toLayer, fromLayer, delayVector, DLB, ldelb)
[DLB,ldelb] = setDLB(1,1,[],DLB,ldelb);
[DLB,ldelb] = setDLB(1,2,[],DLB,ldelb);
[DLB,ldelb] = setDLB(1,3,[],DLB,ldelb);
[DLB,ldelb] = setDLB(2,1,[],DLB,ldelb);         %Normale Forward connection
[DLB,ldelb] = setDLB(2,2,[],DLB,ldelb);
[DLB,ldelb] = setDLB(2,3,[],DLB,ldelb);
[DLB,ldelb] = setDLB(3,2,[],DLB,ldelb);         %Normale Forward connection
[DLB,ldelb] = setDLB(3,3,[],DLB,ldelb);

%Biases
bb(1) = 0;  %0 oder 1
bb(2) = 0;  %0 oder 1
bb(3) = 0;  %0 oder 1














%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             ACHTUNG! ab hier keine �nderungen vornehmen!                %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                              %    %                                     %
%                              %    %                                     %
%                              %    %                                     %
%                              %    %                                     %
%                          %%%%%    %%%%%                                 %
%                            %%%    %%%                                   %
%                              %%  %%                                     %
%                               %%%%                                      %   
%                                %%                                       %
%                                                                         %
%Maximale L�nge der Verz�gerungen
D_max = max(max(max(max(idel))),max(max(max(ldel))));

%Set of input layers
%A layer is an input layer if it has an input weight, of if it contains any
%delays with any of its weight matrices
X = zeros(1,numoflayers);
for i = 1:numoflayers
    for j = 1:numofinputs
        %if it has an input weight
        if DI(i,j) > 0
            X(i) = 1;
        end
    end
end

for i = 1:numoflayers
    for j = 1:numoflayers
        %if it contains any delays with any of its weight matrices
        %if (i > j && DL(i,j) > 1) || (i <= j && DL(i,j) > 0)
        if (DL(i,j) > 1) || (DL(i,j) == 1 && ldel(i,j,1) > 0)
            X(i) = 1;
        end
    end
end

%Set of output layers
%A layer is an output layer if its output will be compared to a target
%during training, or if it is connected to an input layer through a matrix
%that has any delays associated with it
U = U_target;
for i = 1:numoflayers
    for j = 1:numoflayers
        %if (i > j && DL(i,j) > 1) || (i <= j && DL(i,j) > 0)
        if (DL(i,j) > 1) || (DL(i,j) == 1 && ldel(i,j,1) > 0)
            U(j) = 1;
        end
    end
end

%Input connections
I=zeros(1,numoflayers*numofinputs);
for i = 1:numoflayers
    for j = 1:numofinputs
        if DI(i,j) > 0
            I((i-1)*numofinputs+j) = 1;
        end
    end
end

%Forward connections
%Lfm is the set of indices of layers that directly connect forward to layer m
Lf = zeros(1,numoflayers^2);
for i = 1:numoflayers
    for j = 1:numoflayers
        if DL(i,j) > 0
            Lf((i-1)*numoflayers+j) = 1;
        end
    end
end

%Backward connections
%Lbm is the set of indices of layers that are directly connected backwards to layer m
%(or to which layer m connects forward) and that contain no delays in the connection
Lb = zeros(1,numoflayers^2);
for i = 1:numoflayers
    for j = 1:numoflayers
        if DL(i,j) == 1 && ldel(i,j,1) == 0
            Lb((j-1)*numoflayers+i) = 1;
        end
    end
end

%Dimension des Ausgangs der S-Function
S_out = 0;
for i = 1:numoflayers
    if U_out(i) == 1
        S_out = S_out + S(i);
    end
end

%Dimension des zu lernenden Signals und des Fehlers
S_target = 0;
for i = 1:numoflayers
    if U_target(i) ==  1
        S_target = S_target + S(i);
    end
end

%Set E_LW_U(x)
%E_LW_U(x) = {u e U mit der Eigenschaft (LW(x,u) ~= 0)}
E_LW_U = zeros(1,numoflayers^2);
for i = 1:numoflayers
    for j = 1:numoflayers
        if DL(i,j) > 0 && U(j) == 1
            E_LW_U((i-1)*numoflayers+j) = 1;
        end
    end
end

%Matrizen mit L�nge der Delay-Vektoren in Zeilenvektoren umwandeln
for i = 1:numoflayers
    if i == 1
        DI_vec = DI(1,:);
        DL_vec = DL(1,:);
        DIS_vec = DIS(1,:);
        DLS_vec = DLS(1,:);
        DIB_vec = DIB(1,:);
        DLB_vec = DLB(1,:);
    else
        DI_vec = [DI_vec DI(i,:)];
        DL_vec = [DL_vec DL(i,:)];
        DIS_vec = [DIS_vec DIS(i,:)];
        DLS_vec = [DLS_vec DLS(i,:)];
        DIB_vec = [DIB_vec DIB(i,:)];
        DLB_vec = [DLB_vec DLB(i,:)];
    end
end

%Matrizen mit den Delay-Vektoren f�r die Eing�nge in Zeilenvektoren umwandeln
h = 1;
for i = 1:numoflayers
    for j = 1:numofinputs
        for d = 1:DI(i,j)
            idel_vec(h) = idel(i,j,d);
            h = h+1;
        end
    end
end

h = 1;
for i = 1:numoflayers
    for j = 1:numofinputs
        for d = 1:DIS(i,j)
            idels_vec(h) = idels(i,j,d);
            h = h+1;
        end
    end
end

h = 1;
for i = 1:numoflayers
    for j = 1:numofinputs
        for d = 1:DIB(i,j)
            idelb_vec(h) = idelb(i,j,d);
            h = h+1;
        end
    end
end

%Matrizen mit den Delay-Vektoren f�r die Schichten in Zeilenvektoren umwandeln
h = 1;
for i = 1:numoflayers
    for j = 1:numoflayers
        for d = 1:DL(i,j)
            ldel_vec(h) = ldel(i,j,d);
            h = h+1;
        end
    end
end

h = 1;
for i = 1:numoflayers
    for j = 1:numoflayers
        for d = 1:DLS(i,j)
            ldels_vec(h) = ldels(i,j,d);
            h = h+1;
        end
    end
end

h = 1;
for i = 1:numoflayers
    for j = 1:numoflayers
        for d = 1:DLB(i,j)
            ldelb_vec(h) = ldelb(i,j,d);
            h = h+1;
        end
    end
end

if(~exist('idels_vec','var'))
    idels_vec = 0;
end

if(~exist('ldels_vec','var'))
    ldels_vec = 0;
end

if(~exist('idelb_vec','var'))
    idelb_vec = 0;
end

if(~exist('ldelb_vec','var'))
    ldelb_vec = 0;
end

%Matrizen mit den maximalen Delays f�r alle Eing�nge und Schichten ermitteln
dimax = zeros(numoflayers,numofinputs);
dlmax = zeros(numoflayers,numoflayers);
for i = 1:numoflayers
    for j = 1:numofinputs
        dimax(i,j) = max(idel(i,j,:));
    end
    
    for j = 1:numoflayers
        dlmax(i,j) = max(ldel(i,j,:));
    end
end
dimax = max(dimax);
dlmax = max(dlmax);

%Gr��e des Gewichtsvektors
L = 0;
for i = 1:numoflayers
    %Input weights
    for j = 1:numofinputs
        L = L + DI(i,j)*S(i)*R(j);
    end
    
    %Layer weights
    for j = 1:numoflayers
        L = L + DL(i,j)*S(i)*S(j);
    end
    
    %Biases
    L = L + S(i);
end

net_params = [Q, mue, theta, T, L, D_max, numoflayers, numofinputs, max(S), sum(R), S_out, S_target, print_weights, min_weights, prune_interval, train_length, prune_maxerr_OBS, prune_maxerr_exact];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                      Anfangsinitialisierung                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

W_ini = rand(1,L) - 0.5;
W_alpha_ini = ones(1,L);