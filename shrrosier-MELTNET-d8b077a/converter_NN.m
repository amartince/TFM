function [new_structure] = converter_NN(structure)

% Esta funcion cambia los dlarrays presenten en una struct en arrays
% normales para poder leerlos en Julia

% Primero miramos que tipo de estructura le estamos dando a la entrada
if class(structure) == "SeriesNetwork"

    keynames = fieldnames(structure);
    for i = 1:length(keynames)

        value = converter_NN(structure.(keynames{i}));
        new_structure.(keynames{i}) = value;
    end

elseif class(structure) == "nnet.cnn.layer.Layer"

    new_structure = cell(1, length(structure));
    for i = 1:length(structure)

        value = converter_NN(structure(i));
        new_structure{i} = value;
    end

elseif startsWith(class(structure), "nnet.cnn.layer.")

    keynames = fieldnames(structure);
    for i = 1:length(keynames)

        value = converter_NN(structure.(keynames{i}));
        new_structure.(keynames{i}) = value;
    end

elseif class(structure) == "cell"

    new_structure = cell(1, length(structure));
    for i = 1:length(structure)

        value = converter_NN(structure{i});
        new_structure{i} = value;
    end

elseif class(structure) == "dlarray"

    % extractdata transforma un dlarray en un gpuarray
    % gather transforma un gpuarray en un array normal
    new_structure = gather(extractdata(structure));

elseif (class(structure) == "single") || (class(structure) == "char") || (class(structure) == "double") || (class(structure) == "logical")

    % extractdata transforma un dlarray en un gpuarray
    % gather transforma un gpuarray en un array normal
    new_structure = structure;

end

end
