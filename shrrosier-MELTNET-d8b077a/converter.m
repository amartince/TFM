function [new_structure] = converter(structure)

% Esta funcion cambia los dlarrays presenten en una struct en arrays
% normales para poder leerlos en Julia

% Primero miramos que tipo de estructura le estamos dando a la entrada
if class(structure) == "struct"

    keynames = fieldnames(structure);
    for i = 1:length(keynames)

        value = converter(structure.(keynames{i}));
        new_structure.(keynames{i}) = value;
    end

elseif class(structure) == "cell"

    new_structure = cell(1, length(structure));
    for i = 1:length(structure)

        value = converter(structure{i});
        new_structure{i} = value;
    end


elseif class(structure) == "dlarray"

    % extractdata transforma un dlarray en un gpuarray
    % gather transforma un gpuarray en un array normal
    new_structure = gather(extractdata(structure));

end

end

