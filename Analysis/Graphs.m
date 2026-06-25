%%  GRAPHS
close all
clear all
clc

%%  IMPORT DATA
file_path = 'C:\Users\pam1061\OneDrive - USNH\Desktop\Matamala\Thesis\Codes\EcoLab\RE_Journal_update\TopModels\EcoData\St3_Power\DOE_Exp_001_Power.mat';
data_BEM = load(file_path);
w_BEM = data_BEM.frequencies;
RAO_pos_BEM = data_BEM.RAO_with_PTO_relative_abs;
RAO_vel_BEM = w_BEM.*RAO_pos_BEM;
RAO_acc_BEM = w_BEM.^2 .* RAO_pos_BEM;
Power_BEM = data_BEM.P_omega;
load('WECSIM_values.mat');
P_BEM_norm = [185.5, 354.4, 604.2, 931.8, 1278.2, 1524, 1583.6];
P_BEM = P_BEM_norm.*(eta.^2);

%%  GRAPHICS PARAMETERS
TM  = 8;
LW  = 1.4;
Tit = 26;
Ej  = 26;
Let = 26;
Leg = 22;
xmin = 1;
xmax = 5;
hx  = 1;
wmin = 12;
wmax = 36;

%  OUTPUT PATH
out_path = [fileparts(mfilename('fullpath')), '\'];
if ~exist(out_path, 'dir')
    mkdir(out_path);
end

%% -----------------------------------------------------------------------
%%  FIGURE 1: RAO PLOTS
%% -----------------------------------------------------------------------
fig1 = figure(1);
set(fig1, 'Position', [100, 100, 1800, 450]);
set(fig1, 'Color', 'w');

%  SUBPLOT 1: POSITION RAO
subplot(1,3,1)
ymin = 0; ymax = 0.8; hy = 0.2;
hold on
plot(w_BEM(wmin:wmax), RAO_pos_BEM(wmin:wmax), 'k-', 'LineWidth', LW);
plot(w, SD_pos_norm, 'o', 'Color', [0 0.447 0.741], ...
    'MarkerFaceColor', [0 0.447 0.741], 'MarkerSize', TM);
grid(gca, 'minor'); grid on
set(gca, 'YTick', ymin:hy:ymax, 'XTick', xmin:hx:xmax, ...
    'TickLength', [.02 .02], 'XMinorTick', 'on', 'YMinorTick', 'on', ...
    'FontSize', Let, 'FontName', 'Times New Roman', 'LineWidth', 1);
axis([xmin xmax ymin ymax])
xlabel('$\omega$ [rad/s]', 'FontSize', Ej, 'Interpreter', 'latex')
ylabel('$\xi_{rel}/\eta$ [m/m]', 'FontSize', Ej, 'Interpreter', 'latex')
title('(a)', 'FontSize', Tit)

%  SUBPLOT 2: VELOCITY RAO
subplot(1,3,2)
ymin = 0; ymax = 3; hy = 0.5;
hold on
plot(w_BEM(wmin:wmax), RAO_vel_BEM(wmin:wmax), 'k-', 'LineWidth', LW);
plot(w, SD_vel_norm, 'o', 'Color', [0 0.447 0.741], ...
    'MarkerFaceColor', [0 0.447 0.741], 'MarkerSize', TM);
grid(gca, 'minor'); grid on
set(gca, 'YTick', ymin:hy:ymax, 'XTick', xmin:hx:xmax, ...
    'TickLength', [.02 .02], 'XMinorTick', 'on', 'YMinorTick', 'on', ...
    'FontSize', Let, 'FontName', 'Times New Roman', 'LineWidth', 1);
axis([xmin xmax ymin ymax])
xlabel('$\omega$ [rad/s]', 'FontSize', Ej, 'Interpreter', 'latex')
ylabel('$\dot{\xi}_{rel}/\eta$ [1/s]', 'FontSize', Ej, 'Interpreter', 'latex')
title('(b)', 'FontSize', Tit)

%  SUBPLOT 3: POWER
ax3 = subplot(1,3,3);
ymin = 0; ymax = 2; hy = 0.5;
hold on
h1 = plot(w_BEM(wmin:wmax), Power_BEM(wmin:wmax)/1000, 'k-', 'LineWidth', LW);
h2 = plot(w, SD_pow_norm/1000, 'o', 'Color', [0 0.447 0.741], ...
    'MarkerFaceColor', [0 0.447 0.741], 'MarkerSize', TM);
grid(gca, 'minor'); grid on
set(gca, 'YTick', ymin:hy:ymax, 'XTick', xmin:hx:xmax, ...
    'TickLength', [.02 .02], 'XMinorTick', 'on', 'YMinorTick', 'on', ...
    'FontSize', Let, 'FontName', 'Times New Roman', 'LineWidth', 1);
axis([xmin xmax ymin ymax])
xlabel('$\omega$ [rad/s]', 'FontSize', Ej, 'Interpreter', 'latex')
ylabel('$P/\eta^2$ [kW/m$^2$]', 'FontSize', Ej, 'Interpreter', 'latex')
title('(c)', 'FontSize', Tit)
legend(ax3, [h1, h2], 'BEM', 'WECSim', ...
    'FontSize', Leg, 'FontName', 'Times New Roman', ...
    'Orientation', 'vertical', 'Location', 'southeast');

%  GUARDAR FIGURA 1
print(fig1, [out_path, 'Fig1_RAO'], '-dpng', '-r300');
print(fig1, [out_path, 'Fig1_RAO'], '-dpdf');
disp('Fig1 guardada')

%% -----------------------------------------------------------------------
%%  FIGURE 2: BAR CHART - NORMALIZED POWER
%% -----------------------------------------------------------------------
fig2 = figure(2);
set(fig2, 'Position', [100, 100, 1800, 450]);
set(fig2, 'Color', 'w');

%  CONVERSION EFFICIENCY COEFFICIENTS
eff_DDLG = 0.7;
eff_EGEC = 0.55;

%  FREQUENCIES
freqs = [1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0];

%  DATA MATRIX [7x4]
data = [P_BEM_norm', SD_pow_norm', DDLG_pow_norm', EGEC_pow_norm'];

%  COLORS
colors = [
    0.0,  0.0,   0.0;
    0.0,  0.447, 0.741;
    0.85, 0.33,  0.10;
    0.47, 0.67,  0.19;
];

%  BAR PLOT
b = bar(freqs, data, 'grouped', 'BarWidth', 0.92);
hold on

for k = 1:4
    b(k).FaceColor = colors(k,:);
    b(k).EdgeColor = 'k';
    b(k).LineWidth = 0.8;
end

%  ANCHO REAL DE BARRA EN UNIDADES DEL EJE X
bar_w  = (freqs(2) - freqs(1)) * 0.92 / 4;
half_w = bar_w / 2 * 0.95;

%  VALORES DE EFICIENCIA
y_DDLG = P_BEM_norm * eff_DDLG;
y_EGEC = P_BEM_norm * eff_EGEC;

color_DDLG = [0.6 0.1 0.0];
color_EGEC = [0.15 0.45 0.15];

hl1 = NaN; hl2 = NaN;
for i = 1:length(freqs)
    xc_DDLG = b(3).XEndPoints(i);
    xc_EGEC = b(4).XEndPoints(i);
    hl1 = plot([xc_DDLG - half_w, xc_DDLG + half_w], ...
               [y_DDLG(i), y_DDLG(i)], ...
               '-', 'Color', color_DDLG, 'LineWidth', 2.5);
    hl2 = plot([xc_EGEC - half_w, xc_EGEC + half_w], ...
               [y_EGEC(i), y_EGEC(i)], ...
               '-', 'Color', color_EGEC, 'LineWidth', 2.5);
end

%  EJES Y FORMATO
ymin = 0; ymax = 2000; hy = 500;
grid(gca, 'minor'); grid on
set(gca, ...
    'XTick', freqs, 'YTick', ymin:hy:ymax, ...
    'TickLength', [.01 .01], 'XMinorTick', 'off', 'YMinorTick', 'on', ...
    'FontSize', Let, 'FontName', 'Times New Roman', 'LineWidth', 1);
axis([min(freqs)-0.3 max(freqs)+0.3 ymin ymax])
xlabel('$\omega$ [rad/s]', 'FontSize', Ej, 'Interpreter', 'latex')
ylabel('$P/\eta^2$ [W/m$^2$]', 'FontSize', Ej, 'Interpreter', 'latex')

legend([b(1), b(2), b(3), b(4), hl1, hl2], ...
    'Mechanical Power (BEM)', ...
    'Mechanical Power (WECSim)', ...
    'Electrical Power (DDLG)', ...
    'Electrical Power (EGEC)', ...
    ['Expected electrical power DDLG ($\varepsilon_{m \to e}=', num2str(eff_DDLG), '$, Ahamed et al., 2022)'], ...
    ['Expected electrical power EGEC ($\varepsilon_{m \to e}=', num2str(eff_EGEC), '$, Li et al., 2020)'], ...
    'FontSize', Leg, 'FontName', 'Times New Roman', ...
    'Interpreter', 'latex', 'Location', 'northwest');

%  GUARDAR FIGURA 2
print(fig2, [out_path, 'Fig2_PowerBar'], '-dpng', '-r300');
print(fig2, [out_path, 'Fig2_PowerBar'], '-dpdf');
disp('Fig2 guardada')

%% -----------------------------------------------------------------------
%%  FIGURE 3: BAR CHART - ABSOLUTE POWER (sin BEM, sin _norm)
%% -----------------------------------------------------------------------
fig3 = figure(3);
set(fig3, 'Position', [100, 100, 1800, 450]);
set(fig3, 'Color', 'w');

%  DATA MATRIX [7x3] — sin BEM, potencia absoluta
data3 = [SD_pow', DDLG_pow', EGEC_pow'];

%  COLORS — solo 3 barras
colors3 = [
    0.0,  0.447, 0.741;    % Azul  - Mechanical Power (WECSim)
    0.85, 0.33,  0.10;     % Naranja - Electrical Power (DDLG)
    0.47, 0.67,  0.19;     % Verde - Electrical Power (EGEC)
];

%  BAR PLOT
b3 = bar(freqs, data3, 'grouped', 'BarWidth', 0.92);
hold on

for k = 1:3
    b3(k).FaceColor = colors3(k,:);
    b3(k).EdgeColor = 'k';
    b3(k).LineWidth = 0.8;
end

%  LINEAS DE EFICIENCIA — basadas en P_BEM absoluto
bar_w3  = (freqs(2) - freqs(1)) * 0.92 / 3;
half_w3 = bar_w3 / 2 * 0.95;

y_DDLG3 = P_BEM * eff_DDLG;
y_EGEC3 = P_BEM * eff_EGEC;

hl3_1 = NaN; hl3_2 = NaN;
for i = 1:length(freqs)
    xc_DDLG3 = b3(2).XEndPoints(i);
    xc_EGEC3 = b3(3).XEndPoints(i);

    hl3_1 = plot([xc_DDLG3 - half_w3, xc_DDLG3 + half_w3], ...
                 [y_DDLG3(i), y_DDLG3(i)], ...
                 '-', 'Color', color_DDLG, 'LineWidth', 2.5);
    hl3_2 = plot([xc_EGEC3 - half_w3, xc_EGEC3 + half_w3], ...
                 [y_EGEC3(i), y_EGEC3(i)], ...
                 '-', 'Color', color_EGEC, 'LineWidth', 2.5);
end

%  EJES Y FORMATO
ymin = 0; ymax = 40; hy = 10;
grid(gca, 'minor'); grid on
set(gca, ...
    'XTick', freqs, 'YTick', ymin:hy:ymax, ...
    'TickLength', [.01 .01], 'XMinorTick', 'off', 'YMinorTick', 'on', ...
    'FontSize', Let, 'FontName', 'Times New Roman', 'LineWidth', 1);
axis([min(freqs)-0.3 max(freqs)+0.3 ymin ymax])
xlabel('$\omega$ [rad/s]', 'FontSize', Ej, 'Interpreter', 'latex')
ylabel('$P$ [W]', 'FontSize', Ej, 'Interpreter', 'latex')

legend([b3(1), b3(2), b3(3), hl3_1, hl3_2], ...
    'Mechanical Power (WECSim)', ...
    'Electrical Power (DDLG)', ...
    'Electrical Power (EGEC)', ...
    ['Expected electrical power DDLG ($\varepsilon_{m \to e}=', num2str(eff_DDLG), '$, Ahamed et al., 2022)'], ...
    ['Expected electrical power EGEC ($\varepsilon_{m \to e}=', num2str(eff_EGEC), '$, Li et al., 2020)'], ...
    'FontSize', Leg, 'FontName', 'Times New Roman', ...
    'Interpreter', 'latex', 'Location', 'northwest');

%  GUARDAR FIGURA 3
print(fig3, [out_path, 'Fig3_PowerBar_Abs'], '-dpng', '-r300');
print(fig3, [out_path, 'Fig3_PowerBar_Abs'], '-dpdf');
disp('Fig3 guardada')

disp('--- Todas las figuras guardadas en:')
disp(out_path)

%% -----------------------------------------------------------------------
%%  FIGURE 4: ABSOLUTE POWER + WAVE AMPLITUDE (yyaxis)
%% -----------------------------------------------------------------------
fig4 = figure(4);
set(fig4, 'Position', [100, 100, 1800, 600]);
set(fig4, 'Color', 'w');

%  AMPLITUDES DE OLA
eta_vals = [0.05, 0.07, 0.09, 0.12, 0.14, 0.16, 0.16];

%  DATA MATRIX [7x3]
data4 = [SD_pow', DDLG_pow', EGEC_pow'];

%  COLORS
colors4 = [
    0.0,  0.447, 0.741;
    0.85, 0.33,  0.10;
    0.47, 0.67,  0.19;
];

color_DDLG = [0.6 0.1 0.0];
color_EGEC = [0.15 0.45 0.15];

%  EJE IZQUIERDO - POTENCIA
yyaxis left

%  BAR PLOT
b4 = bar(freqs, data4, 'grouped', 'BarWidth', 0.92);
hold on

for k = 1:3
    b4(k).FaceColor = colors4(k,:);
    b4(k).EdgeColor = 'k';
    b4(k).LineWidth = 0.8;
end

%  LINEAS DE EFICIENCIA
bar_w4  = (freqs(2) - freqs(1)) * 0.92 / 3;
half_w4 = bar_w4 / 2 * 0.95;

y_DDLG4 = P_BEM * eff_DDLG;
y_EGEC4 = P_BEM * eff_EGEC;

hl4_1 = NaN; hl4_2 = NaN;
for i = 1:length(freqs)
    xc_DDLG4 = b4(2).XEndPoints(i);
    xc_EGEC4 = b4(3).XEndPoints(i);
    hl4_1 = plot([xc_DDLG4 - half_w4, xc_DDLG4 + half_w4], ...
                 [y_DDLG4(i), y_DDLG4(i)], ...
                 '-', 'Color', color_DDLG, 'LineWidth', 2.5);
    hl4_2 = plot([xc_EGEC4 - half_w4, xc_EGEC4 + half_w4], ...
                 [y_EGEC4(i), y_EGEC4(i)], ...
                 '-', 'Color', color_EGEC, 'LineWidth', 2.5);
end

%  FORMATO EJE IZQUIERDO
ylabel('$P$ [W]', 'FontSize', Ej, 'Interpreter', 'latex')
set(gca, ...
    'YColor',      'k', ...
    'YLim',        [0 40], ...
    'YTick',       0:10:40, ...
    'YMinorTick',  'on')    % <-- ESTA ES LA LINEA QUE FALTA

%  EJE DERECHO - AMPLITUD DE OLA
yyaxis right

heta = plot(freqs, eta_vals, '-o', ...
    'Color',           'k', ...
    'LineWidth',       2.5, ...
    'MarkerSize',      9, ...
    'MarkerFaceColor', 'k');

ylabel('$\eta$ [m]', 'FontSize', Ej, 'Interpreter', 'latex', ...
    'Color', 'k')
set(gca, 'YColor', 'k', 'YLim', [0 0.25], 'YTick', 0:0.05:0.25)

%  FORMATO GENERAL
grid(gca, 'minor'); grid on
set(gca, ...
    'XTick',      freqs, ...
    'TickLength', [.01 .01], ...
    'XMinorTick', 'off', ...
    'YMinorTick', 'on', ...
    'FontSize',   Let, ...
    'FontName',   'Times New Roman', ...
    'LineWidth',  1, ...
    'XLim',       [min(freqs)-0.3, max(freqs)+0.3]);

xlabel('$\omega$ [rad/s]', 'FontSize', Ej, 'Interpreter', 'latex')

%  ENTRADAS FANTASMA PARA TITULOS - creadas en yyaxis right
yyaxis right
h_title1 = plot(NaN, NaN, 'w', 'LineWidth', 0.01);
h_title2 = plot(NaN, NaN, 'w', 'LineWidth', 0.01);

%  LEYENDA
lgd = legend([h_title1, b4(1), b4(2), b4(3), hl4_1, hl4_2, h_title2, heta], ...
    '\bf LEFT AXIS $-$ $P$ [W]', ...
    'Mech. Power (SD)', ...
    'Elec. Power (DDLG)', ...
    'Elec. Power (EGEC)', ...
    ['Exp. elec. power (DDLG)'], ...
    ['Exp. elec. power (EGEC)'], ...
    '\bf RIGHT AXIS $-$ $\eta$ [m]', ...
    'Wave amplitude', ...
    'FontSize', Leg, 'FontName', 'Times New Roman', ...
    'Interpreter', 'latex', ...
    'Location', 'northwest', ...
    'Box', 'on');

%  TAMAÑO Y POSICION MANUAL
%  [x_izquierda  y_abajo  ancho  alto]
lgd.Position = [0.15, 0.55, 0.24, 0.36];
%lgd.ItemTokenSize = [15, 5];   % reduce el ancho del icono de color en la leyenda

%  GUARDAR FIGURA 4
exportgraphics(fig4, [out_path, 'Fig4_PowerBar_Eta.pdf'], ...
    'ContentType', 'vector', 'BackgroundColor', 'white');
print(fig4, [out_path, 'Fig4_PowerBar_Eta'], '-dpng', '-r300');
disp('Fig4 guardada')

disp('--- Todas las figuras guardadas en:')
disp(out_path)