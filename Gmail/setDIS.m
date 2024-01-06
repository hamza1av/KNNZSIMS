function [DIS, idels] = setDIS(toLayer, fromLayer, delayVector, DIS, idels)
%function [DIS, idels] = setDIS(toLayer, fromLayer, delayVector, DIS, idels)

%Funktion zum Belegen von DIS(toLayer, fromLayer) mit der Länge von delayVector
%und idels(toLayer, fromLayer) mit delayVector

delayVector = sort(delayVector);

DIS(toLayer, fromLayer) = length(delayVector);

for i = 1:DIS(toLayer, fromLayer)
    idels(toLayer, fromLayer, i) = delayVector(i);
end
