function tutorial_entrenamiento()
%--------------------------------------------------------------------------
% Ultima actualizacion Novimebre 2020 por Sergio Ramirez
%--------------------------------------------------------------------------

% -------------------------------------------------------------------------
% Ensemble Classification | June 2013 | version 2.0 | TUTORIAL
% -------------------------------------------------------------------------
% Copyright (c) 2013 DDE Lab, Binghamton University, NY.
% All Rights Reserved.
% -------------------------------------------------------------------------
% Contact: jan@kodovsky.com | fridrich@binghamton.edu | June 2013
%          http://dde.binghamton.edu/download/ensemble
% -------------------------------------------------------------------------

% Short tutorial for the ensemble classifier (ver 2.0) developed for
% steganalysis in digital images.

% Cargar caracterisiticas.
cover = load('/rutacover.mat');
stego = load('/rutastego.mat');

% Ambos archivos tienen una matriz de carateristicas 'F', donde cada fila
% corresponde al un vector y el numero de coloumnas es la dimencionalidad
% del vector. Contiene una estructura 'names' con el nombre de la imagen
% correspondiente al vector de caracteristicas.

% Los nombres de stego/cover.names estan desincronizados. La
% sincronizacion es importante como un paso de preprocesamiento en
% estegoanalsis. Po lo tanto, deben de sincronizarse en pares. Vector de
% imagen cover => vector imagen stego

%{
En el esteganálisis, es importante entrenar en los * pares * de 
caracteristicas cover y las características  stego correspondientes.
Al dividir las características en partes de entrenamiento / prueba, por 
ejemplo, estos pares deben conservarse. ¡Pero es igualmente importante 
hacer un seguimiento de estos pares incluso dentro del conjunto de 
entrenamiento! Esto se debe a que al dividir los datos de entrenamiento 
con fines de validación cruzada (o bootstrapping), estos pares, nuevamente,
necesitan ser preservados. Por lo tanto, nuestra implementación de la 
formación de conjunto acepta solo dos matrices * sincronizadas * de 
características (cover y stego). Mientras que la implementación verifica 
los tamaños de ambas matrices, la sincronización real es responsabilidad de
un usuario. Vea el siguiente código como ejemplo de cómo hacer esto 
correctamente. 
%}

% Restriccion: solo se consideran las carateristicas que tienen cover y su
% correspondientes stego, elimina los no pares c=>s
names = intersect(cover.names,stego.names); % cell array
names = sort(names); % ordena los nombres

% Preparar caracteristicas cover C
% ismember(A,B) retorna vector A con 1 cuando es miembro B, 0 si no 
cover_names = cover.names(ismember(cover.names,names));
% ix retona index de las posiciones iniciales
[cover_names,ix] = sort(cover_names);

% ismem retorna vector de 111..., obtienen los vectores
C = cover.F(ismember(cover.names,names),:);
C = C(ix,:); % aplica sort de ix

% Preparar carateriticas stego S
stego_names = stego.names(ismember(stego.names,names));
[stego_names,ix] = sort(stego_names);
S = stego.F(ismember(stego.names,names),:);
S = S(ix,:); % aplica sort de ix
%fprintf('size S:');
%disp(size(S))

%{
En este punto, tenemos las características cover C y las 
características stego correspondientes S. Están sincronizadas 
correctamente, es decir, la i-ésima fila de la matriz stego S proviene 
de la imagen stego que se creó a partir de la imagen cover con 
características en el i-ésima fila de la matriz cover C.
%}

tic
settings = struct('verbose',2);
for seed = 1:10
    RandStream.setGlobalStream(RandStream('mt19937ar','Seed',seed));
    random_permutation = randperm(size(C,1));% regresa vector aleatorio de N
    training_set = random_permutation(1:5000);%(1:round(size(C,1)/2));
    testing_set = random_permutation(5000+1:end);%(round(size(C,1)/2)+1:end);
    
    %guardando los vectores de prueba, cover y stego
    save(strcat('test_set',int2str(seed),'.mat'),'testing_set');
    
    TRN_cover = C(training_set,:); % devuelve matriz de vectores
    TRN_stego = S(training_set,:);
    
    % entrena el clasificador y regresa vector de pesos
    [trained_ensemble,results] = ensemble_training(TRN_cover,TRN_stego,settings);
    % guarda el entrenamiento
    save(strcat('trained_e',int2str(seed),'.mat'),'trained_ensemble','results');
    clearvars TRN_cover TRN_stego
    
end
toc