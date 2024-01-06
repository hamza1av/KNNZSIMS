function [DIB, idelb] = setDIB(toLayer, fromLayer, delayVector, DIB, idelb)
%function [DIB, idelb] = setDIB(toLayer, fromLayer, delayVector, DIB, idelb)

%Funktion zum Belegen von DIB(toLayer, fromLayer) mit der Länge von delayVector
%und idelb(toLayer, fromLayer) mit delayVector

delayVector = sort(delayVector);

DIB(toLayer, fromLayer) = length(delayVector);

for i = 1:DIB(toLayer, fromLayer)
    idelb(toLayer, fromLayer, i) = delayVector(i);
end
