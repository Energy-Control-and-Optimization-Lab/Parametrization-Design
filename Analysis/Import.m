%%  DATA ANALYSIS

close all
clear all
clc

%%  INPUT DATA - LOOP OVER ALL SD CONDITIONS

basePath = 'C:\Users\pam1061\OneDrive - USNH\Desktop\Matamala\Thesis\Codes\EcoLab\RE_Journal_update\WECSIM\PTOs\MECC2026\SD\';

conditions = {'SD_w16_h010', ...
              'SD_w20_h014', ...
              'SD_w24_h018', ...
              'SD_w28_h024', ...
              'SD_w32_h028', ...
              'SD_w36_h032', ...
              'SD_w40_h032'};

for i = 1:length(conditions)
    
    cond = conditions{i};
    filePath = fullfile(basePath, cond, 'output', 'ECO_RM_WEC_SD_matlabWorkspace.mat');
    
    fprintf('Loading: %s\n', cond);
    load(filePath);
    
    %  EXTRACT PTO HEAVE DATA (column 3) AS VECTORS
    time                         = output.ptos(1).time(:);
    position_heave               = output.ptos(1).position(:,3);
    velocity_heave               = output.ptos(1).velocity(:,3);
    acceleration_heave           = output.ptos(1).acceleration(:,3);
    powerInternalMechanics_heave = output.ptos(1).powerInternalMechanics(:,3);
    
    %  STORE IN NAMED VARIABLE (e.g. SD_w16_h010)
    eval([cond ' = [time position_heave velocity_heave acceleration_heave powerInternalMechanics_heave];']);
    
end
disp('Variables: SD_w16_h010, SD_w20_h014, SD_w24_h018, SD_w28_h024, SD_w32_h028, SD_w36_h032, SD_w40_h032')

%%  INPUT DATA - LOOP OVER ALL DDLG CONDITIONS

basePath = 'C:\Users\pam1061\OneDrive - USNH\Desktop\Matamala\Thesis\Codes\EcoLab\RE_Journal_update\WECSIM\PTOs\MECC2026\DDLG\';

conditions = {'DDLG_w16_h010', ...
              'DDLG_w20_h014', ...
              'DDLG_w24_h018', ...
              'DDLG_w28_h024', ...
              'DDLG_w32_h028', ...
              'DDLG_w36_h032', ...
              'DDLG_w40_h032'};

for i = 1:length(conditions)
    
    cond = conditions{i};
    filePath = fullfile(basePath, cond, 'output', 'ECO_RM_WEC__matlabWorkspace.mat');
    
    fprintf('Loading: %s\n', cond);
    load(filePath);
    
    %  EXTRACT PTO HEAVE DATA (column 3) AS VECTORS
    time                         = output.ptoSim(1).time(:);
    elecPower_heave = output.ptoSim(1).elecPower(:);
    
    %  STORE IN NAMED VARIABLE (e.g. SD_w16_h010)
    eval([cond ' = [time elecPower_heave];']);
    
end

disp('Variables: DDLGw16_h010, DDLGw20_h014, DDLGw24_h018, DDLGw28_h024, DDLGw32_h028, DDLGw36_h032, DDLGw40_h032')

%%  INPUT DATA - LOOP OVER ALL EGEC CONDITIONS

basePath = 'C:\Users\pam1061\OneDrive - USNH\Desktop\Matamala\Thesis\Codes\EcoLab\RE_Journal_update\WECSIM\PTOs\MECC2026\EGEC\';

conditions = {'EGEC_w16_h010', ...
              'EGEC_w20_h014', ...
              'EGEC_w24_h018', ...
              'EGEC_w28_h024', ...
              'EGEC_w32_h028', ...
              'EGEC_w36_h032', ...
              'EGEC_w40_h032'};

for i = 1:length(conditions)
    
    cond = conditions{i};
    filePath = fullfile(basePath, cond, 'output', 'ECO_RM_WEC__matlabWorkspace.mat');
    
    fprintf('Loading: %s\n', cond);
    load(filePath);
    
    %  EXTRACT PTO HEAVE DATA (column 3) AS VECTORS
    time                         = output.ptoSim(1).time(:);
    elecPower_heave = output.ptoSim(1).P_elec(:);
    
    %  STORE IN NAMED VARIABLE (e.g. SD_w16_h010)
    eval([cond ' = [time elecPower_heave];']);
    
end

disp('Variables: EGECw16_h010, EGECw20_h014, EGECw24_h018, EGECw28_h024, EGECw32_h028, EGECw36_h032, EGECw40_h032')

%%  SAVE DATA
save('WECSIM_vectors','SD_w16_h010','SD_w20_h014','SD_w24_h018','SD_w28_h024','SD_w32_h028','SD_w36_h032','SD_w40_h032',...
    'DDLG_w16_h010','DDLG_w20_h014','DDLG_w24_h018','DDLG_w28_h024','DDLG_w32_h028','DDLG_w36_h032','DDLG_w40_h032',...
    'EGEC_w16_h010','EGEC_w20_h014','EGEC_w24_h018','EGEC_w28_h024','EGEC_w32_h028','EGEC_w36_h032','EGEC_w40_h032')

%%  RAOS
data_WECSIM = load('WECSIM_vectors.mat');
t   = data_WECSIM.DDLG_w16_h010(:,1);
rampTime = 30;
idx = t >= rampTime;
H = [0.1 0.14 0.18 0.24 0.28 0.32 0.32];
eta = 0.5.*H;
w = [1.6 2.0 2.4 2.6 3.0 3.6 4.0];

%Position
% Caso 1
signal = abs(data_WECSIM.SD_w16_h010(idx,2));
[pks,~] = findpeaks(signal);
SD_pos(1) = mean(pks);
% Caso 2
signal = abs(data_WECSIM.SD_w20_h014(idx,2));
[pks,~] = findpeaks(signal);
SD_pos(2) = mean(pks);
% Caso 3
signal = abs(data_WECSIM.SD_w24_h018(idx,2));
[pks,~] = findpeaks(signal);
SD_pos(3) = mean(pks);
% Caso 4
signal = abs(data_WECSIM.SD_w28_h024(idx,2));
[pks,~] = findpeaks(signal);
SD_pos(4) = mean(pks);
% Caso 5
signal = abs(data_WECSIM.SD_w32_h028(idx,2));
[pks,~] = findpeaks(signal);
SD_pos(5) = mean(pks);
% Caso 6
signal = abs(data_WECSIM.SD_w36_h032(idx,2));
[pks,~] = findpeaks(signal);
SD_pos(6) = mean(pks);
% Caso 7
signal = abs(data_WECSIM.SD_w40_h032(idx,2));
[pks,~] = findpeaks(signal);
SD_pos(7) = mean(pks);

SD_pos_norm = SD_pos ./ eta;

%Velocity
signal = abs(data_WECSIM.SD_w16_h010(idx,3));
[pks,~] = findpeaks(signal);
SD_vel(1) = mean(pks);
% Caso 2
signal = abs(data_WECSIM.SD_w20_h014(idx,3));
[pks,~] = findpeaks(signal);
SD_vel(2) = mean(pks);
% Caso 3
signal = abs(data_WECSIM.SD_w24_h018(idx,3));
[pks,~] = findpeaks(signal);
SD_vel(3) = mean(pks);
% Caso 4
signal = abs(data_WECSIM.SD_w28_h024(idx,3));
[pks,~] = findpeaks(signal);
SD_vel(4) = mean(pks);
% Caso 5
signal = abs(data_WECSIM.SD_w32_h028(idx,3));
[pks,~] = findpeaks(signal);
SD_vel(5) = mean(pks);
% Caso 6
signal = abs(data_WECSIM.SD_w36_h032(idx,3));
[pks,~] = findpeaks(signal);
SD_vel(6) = mean(pks);
% Caso 7
signal = abs(data_WECSIM.SD_w40_h032(idx,3));
[pks,~] = findpeaks(signal);
SD_vel(7) = mean(pks);

SD_vel_norm = SD_vel ./ eta;

%Acceleration
signal = abs(data_WECSIM.SD_w16_h010(idx,4));
[pks,~] = findpeaks(signal);
SD_acc(1) = mean(pks);
% Caso 2
signal = abs(data_WECSIM.SD_w20_h014(idx,4));
[pks,~] = findpeaks(signal);
SD_acc(2) = mean(pks);
% Caso 3
signal = abs(data_WECSIM.SD_w24_h018(idx,4));
[pks,~] = findpeaks(signal);
SD_acc(3) = mean(pks);
% Caso 4
signal = abs(data_WECSIM.SD_w28_h024(idx,4));
[pks,~] = findpeaks(signal);
SD_acc(4) = mean(pks);
% Caso 5
signal = abs(data_WECSIM.SD_w32_h028(idx,4));
[pks,~] = findpeaks(signal);
SD_acc(5) = mean(pks);
% Caso 6
signal = abs(data_WECSIM.SD_w36_h032(idx,4));
[pks,~] = findpeaks(signal);
SD_acc(6) = mean(pks);
% Caso 7
signal = abs(data_WECSIM.SD_w40_h032(idx,4));
[pks,~] = findpeaks(signal);
SD_acc(7) = mean(pks);

SD_acc_norm = SD_acc ./ eta;

%Power
SD_pow(1) = abs(mean(data_WECSIM.SD_w16_h010(idx,5)));
SD_pow(2) = abs(mean(data_WECSIM.SD_w20_h014(idx,5)));
SD_pow(3) = abs(mean(data_WECSIM.SD_w24_h018(idx,5)));
SD_pow(4) = abs(mean(data_WECSIM.SD_w28_h024(idx,5)));
SD_pow(5) = abs(mean(data_WECSIM.SD_w32_h028(idx,5)));
SD_pow(6) = abs(mean(data_WECSIM.SD_w36_h032(idx,5)));
SD_pow(7) = abs(mean(data_WECSIM.SD_w40_h032(idx,5)));

SD_pow_norm = SD_pow ./ (eta.^2);

DDLG_pow(1) = mean(data_WECSIM.DDLG_w16_h010(idx,2));
DDLG_pow(2) = mean(data_WECSIM.DDLG_w20_h014(idx,2));
DDLG_pow(3) = mean(data_WECSIM.DDLG_w24_h018(idx,2));
DDLG_pow(4) = mean(data_WECSIM.DDLG_w28_h024(idx,2));
DDLG_pow(5) = mean(data_WECSIM.DDLG_w32_h028(idx,2));
DDLG_pow(6) = mean(data_WECSIM.DDLG_w36_h032(idx,2));
DDLG_pow(7) = mean(data_WECSIM.DDLG_w40_h032(idx,2));

DDLG_pow_norm = DDLG_pow ./ (eta.^2);

EGEC_pow(1) = mean(data_WECSIM.EGEC_w16_h010(idx,2));
EGEC_pow(2) = mean(data_WECSIM.EGEC_w20_h014(idx,2));
EGEC_pow(3) = mean(data_WECSIM.EGEC_w24_h018(idx,2));
EGEC_pow(4) = mean(data_WECSIM.EGEC_w28_h024(idx,2));
EGEC_pow(5) = mean(data_WECSIM.EGEC_w32_h028(idx,2));
EGEC_pow(6) = mean(data_WECSIM.EGEC_w36_h032(idx,2));
EGEC_pow(7) = mean(data_WECSIM.EGEC_w40_h032(idx,2));

EGEC_pow_norm = EGEC_pow ./ (eta.^2);

save('WECSIM_values','SD_pos','SD_pos_norm','SD_vel','SD_vel_norm','SD_acc','SD_acc_norm','SD_pow','SD_pow_norm','DDLG_pow_norm','DDLG_pow','EGEC_pow_norm','EGEC_pow','H','eta','w')
