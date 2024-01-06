function [DL, ldel] = setDL(toLayer, fromLayer, delayVector, DL, ldel)
%function [DL, ldel] = setDL(toLayer, fromLayer, delayVector, DL, ldel)

%Funktion zum Belegen von DL(toLayer, fromLayer) mit der Länge von delayVector
%und ldel(toLayer, fromLayer) mit delayVector

delayVector = sort(delayVector);

DL(toLayer, fromLayer) = length(delayVector);

for i = 1:DL(toLayer, fromLayer)
    ldel(toLayer, fromLayer, i) = delayVector(i);
end
