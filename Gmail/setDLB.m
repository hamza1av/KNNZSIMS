function [DLB, idelb] = setDLB(toLayer, fromLayer, delayVector, DLB, idelb)
%function [DLB, idelb] = setDLB(toLayer, fromLayer, delayVector, DLB, idelb)

%Funktion zum Belegen von DLB(toLayer, fromLayer) mit der Länge von delayVector
%und idelb(toLayer, fromLayer) mit delayVector

delayVector = sort(delayVector);

DLB(toLayer, fromLayer) = length(delayVector);

for i = 1:DLB(toLayer, fromLayer)
    idelb(toLayer, fromLayer, i) = delayVector(i);
end
