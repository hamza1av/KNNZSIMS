function [DLS, ldels] = setDLS(toLayer, fromLayer, delayVector, DLS, ldels)
%function [DLS, ldels] = setDLS(toLayer, fromLayer, delayVector, DLS, ldels)

%Funktion zum Belegen von DLS(toLayer, fromLayer) mit der Länge von delayVector
%und ldels(toLayer, fromLayer) mit delayVector

delayVector = sort(delayVector);

DLS(toLayer, fromLayer) = length(delayVector);

for i = 1:DLS(toLayer, fromLayer)
    ldels(toLayer, fromLayer, i) = delayVector(i);
end
