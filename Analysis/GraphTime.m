%%  TIME SERIES - w40 h032
close all
clc

%%  LOAD DATA
load('WECSIM_vectors.mat');

%%  EXTRACTION
%%  EXTRACTION
t_SD   = SD_w40_h032(:,1);
P_SD   = abs(SD_w40_h032(:,5));

t_DDLG = DDLG_w40_h032(:,1);
P_DDLG = DDLG_w40_h032(:,2);

t_EGEC = EGEC_w40_h032(:,1);
P_EGEC = EGEC_w40_h032(:,2);

%%  RAMP TIME
t_ramp = 30;   % [s] — restar media solo a partir de aqui

%%  RESTAR MEDIA POST-RAMP
P_SD   = P_SD   - mean(P_SD(t_SD     >= t_ramp));
P_DDLG = P_DDLG - mean(P_DDLG(t_DDLG >= t_ramp));
P_EGEC = P_EGEC - mean(P_EGEC(t_EGEC >= t_ramp));

%%  EFFICIENCY COEFFICIENTS
eff_DDLG = 0.70;
eff_EGEC = 0.55;

%%  APPLY EFFICIENCY
P_DDLG_elec = P_DDLG * eff_DDLG;
P_EGEC_elec = P_EGEC * eff_EGEC;

%%  TIME RANGE — ajusta estos valores
t_start = 60;     % [s]
t_end   = 70;    % [s]

%%  FILTER TIME RANGE
idx_SD   = t_SD   >= t_start & t_SD   <= t_end;
idx_DDLG = t_DDLG >= t_start & t_DDLG <= t_end;
idx_EGEC = t_EGEC >= t_start & t_EGEC <= t_end;

%%  GRAPHICS PARAMETERS
LW  = 1.4;

Ej  = 26;
Let = 26;
Leg = 22;

%%  OUTPUT PATH
out_path = [fileparts(mfilename('fullpath')), '\'];
if ~exist(out_path, 'dir')
    mkdir(out_path);
end

%%  FIGURE 5: TIME SERIES
fig5 = figure(5);
set(fig5, 'Position', [100, 100, 1200, 650]);
set(fig5, 'Color', 'w');

hold on

h_SD   = plot(t_SD(idx_SD),   P_SD(idx_SD),   '-', ...
    'Color', [0.0, 0.25, 0.55], 'LineWidth', 2.0);   % azul oscuro
h_DDLG = plot(t_DDLG(idx_DDLG), P_DDLG_elec(idx_DDLG), '-', ...
    'Color', [0.65 0.15 0.02],   'LineWidth', 2.0);   % naranja oscuro
h_EGEC = plot(t_EGEC(idx_EGEC), P_EGEC_elec(idx_EGEC), '-', ...
    'Color', [0.20 0.45 0.08],   'LineWidth', 2.0);   % verde oscuro

grid(gca, 'minor'); grid on
set(gca, ...
    'TickLength', [.01 .01], ...
    'XMinorTick', 'on', ...
    'YMinorTick', 'on', ...
    'FontSize',   Let, ...
    'FontName',   'Times New Roman', ...
    'LineWidth',  1);

xlabel('$t$ [s]',  'FontSize', Ej, 'Interpreter', 'latex')
ylabel('$P(t)$ [W]',  'FontSize', Ej, 'Interpreter', 'latex')

lgd = legend([h_SD, h_DDLG, h_EGEC], ...
    'Mech. Power (SD)', ...
    ['Exp. Elec. Power (DDLG)'], ...
    ['Exp. Elec. Power (EGEC)'], ...
    'FontSize', Leg, 'FontName', 'Times New Roman', ...
    'Interpreter', 'latex', 'Location', 'southeast');

lgd.Position = [0.53, 0.86, 0.38, 0.04];   % [x  y  ancho  alto]
lgd.ItemTokenSize = [50, 20];                % reduce espacio del icono


%%  LIMITES EJE Y
ymin_ts = -50;   % ajusta segun tus datos
ymax_ts =  50;   % ajusta segun tus datos

xlim([t_start t_end])
ylim([ymin_ts ymax_ts])

%%  GUARDAR
exportgraphics(fig5, [out_path, 'Fig5_TimeSeries.pdf'], ...
    'ContentType', 'vector', 'BackgroundColor', 'white');
print(fig5, [out_path, 'Fig5_TimeSeries'], '-dpng', '-r300');
disp('Fig5 guardada')