function [DI, idel] = setDI(toLayer, fromLayer, delayVector, DI, idel)
%function [DI, idel] = setDI(toLayer, fromLayer, delayVector, DI, idel)

%Funktion zum Belegen von DI(toLayer, fromLayer) mit der Länge von delayVector
%und idel(toLayer, fromLayer) mit delayVector

delayVector = sort(delayVector);

DI(toLayer, fromLayer) = length(delayVector);

for i = 1:DI(toLayer, fromLayer)
    idel(toLayer, fromLayer, i) = delayVector(i);
end
