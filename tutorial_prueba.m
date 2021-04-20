function tutorial_prueba()
%--------------------------------------------------------------------------
% Ultima actulazacion Novimebre 2020 por Sergio Ramirez
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

% se cargan las carateristicas {F,names}
cover = load('/ruta/cover.mat');
stego = load('/ruta/stego.mat'); 
% se asume que ambas caracteristicas, son simetricas

% Preparar caracteristicas cover C
cover_names = cover.names;
% ix retona idn de las posiciones iniciales
[cover_names,ix] = sort(cover_names);

C = cover.F; % obtiene todos los vectores
C = C(ix,:); % aplica sort de ix

% Preparar caracteristicas Stego S
stego_names = stego.names;
% ix retona idn de las posiciones iniciales
[stego_names,ix] = sort(stego_names);

S = stego.F; % obtiene todos los vectores
S = S(ix,:); % aplica sort de ix

fprintf('Pruebas\n')
tic
%metricas
testing_errors = zeros(1,10);
accuracy_errors = zeros(1,10);
f_measure_erros = zeros(1,10);
for semilla = 1:10
    % lee caraterirticas para prueba (indices)
    TEST_ind = load(strcat('test_set',int2str(semilla),'.mat'));
    
    TEST_stego = S(TEST_ind.testing_set,:);
    TEST_cover = C(TEST_ind.testing_set,:);
    % lee el clasificador 
    ENS = load(strcat('trained_e',int2str(semilla),'.mat'));
    
    test_results_cover = ensemble_testing(TEST_cover,ENS.trained_ensemble);
    test_results_stego = ensemble_testing(TEST_stego,ENS.trained_ensemble);
    
    % suma la cantidad de predicciones cuando son difererntes que -1
    % false positive FP
    false_alarms = sum(test_results_cover.predictions~=-1);
    % true negative TN
    true_negative = sum(test_results_cover.predictions~=+1); 
    % false negative FN
    missed_detections = sum(test_results_stego.predictions~=+1);
    % true positive TP
    true_positive = sum(test_results_stego.predictions~=-1); 
    % matriz de confusion
    fprintf('\n=====>  Stego | Cover \n');
    fprintf('Stego | %i  : %i\n',true_positive,false_alarms);
    fprintf('Cover | %i  : %i\n',missed_detections,true_negative);
    
    num_testing_samples = size(TEST_cover,1)+size(TEST_stego,1);
    % error PE
    testing_errors(semilla) = (false_alarms + missed_detections)/num_testing_samples;
    % accuracy
    accuracy_errors(semilla) = (true_positive + true_negative)/num_testing_samples;
    % F-measure
    f_measure_erros(semilla) = (2*true_positive)/(2*true_positive+false_alarms+missed_detections);
    fprintf('\n Accuracy : %.4f\n',accuracy_errors(semilla));
    fprintf(' F-measure : %.4f\n',f_measure_erros(semilla));
    fprintf(' Testing error %i: %.4f\n',semilla,testing_errors(semilla));
    fprintf('---------------------------\n')
end
fprintf('\nAverage testing PE error over 10 splits: %.4f (+/- %.4f)\n',mean(testing_errors),std(testing_errors));
fprintf('Average Accuracy error over 10 splits: %.4f (+/- %.4f)\n',mean(accuracy_errors),std(accuracy_errors));
fprintf('Average F-measure error over 10 splits: %.4f (+/- %.4f)\n',mean(f_measure_erros),std(f_measure_erros));
toc
clearvars 