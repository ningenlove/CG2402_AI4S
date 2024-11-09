function plot_slice

format short e
close all
path = 'Result/simulation_0';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%------------------- network solution at t1 -------------------%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% slice of network solution in 2d
load(strcat(path, '/u_pred_t1'))
Z = u_pred_t1;

[X, Y] = meshgrid(-1:0.007:6, 0:0.001:1);

Z = reshape(Z, 1001, 1001);

figure
surf(X, Y, Z)
shading interp
alpha 0.5
set(0, 'defaultfigurecolor', 'w')
xlim([-1, 6])
ylim([0, 1])
pos = axis;
xlabel('$x$','Interpreter','latex', 'Position', [(pos(2)+pos(1))/2, pos(3)-0.15])
ylabel('$\varphi$','Interpreter','latex', 'Position', [pos(1)-0.65, (pos(3) + pos(4))/2])  
zlabel('$\hat{u}(x,t_1,\varphi)$','Interpreter','latex');
set(gca,'FontSize',18);
hold on

X_line = -1:0.007:6;
T_line = 1.5;
Y_line = 0;
Y_line = Y_line + heaviside(X_line - T_line);

% plot function line
load(strcat(path, '/u_NN_t1'))
u_NN = reshape(u_NN_t1, size(X_line));
M = max(u_NN);
index = find(u_NN >= M - 0.01);
u_NN(min(index)) = nan;
u_NN(max(index)) = nan;
h = plot3(X_line, Y_line, u_NN, 'k', 'LineWidth', 3);
legend(h, {'$\hat{u}(x,t_1,\varphi(x,t_1))$'},'Interpreter','latex','Location','northoutside') 
colorbar

view([-21, 76])

ax = gcf;
exportgraphics(ax, 'Figures/Lifted_uNN_t1.pdf')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load(strcat(path, '/u_NN_t1'))
load(strcat(path, '/u_exact_t1'))
u_Exact = u_Exact_t1;
u_NN1 = u_NN_t1;
x = -1 : 0.007 : 6;

figure
plot(x, u_Exact, 'LineWidth', 2, 'color','red')
hold on
xlim([-1, 6])
plot(x, u_NN1, '--k', 'LineWidth', 2)
legend({'$u(x,t_1)$','$\check{u}(x,t_1)$'},'Interpreter','latex') 

set(0,'defaultfigurecolor','w')
xlim([-1, 6])
xlabel('$x$','Interpreter','latex')
ylim([-0.2 1.2])
ylabel('solution value','Interpreter','latex')
set(gca,'FontSize',18);

ax = gcf;
exportgraphics(ax, 'Figures/Projected_uNN_t1.pdf')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure
load(strcat(path, '/pterr_t1'))
u_err = pterr_t1;
plot(x, log10(abs(u_err)), 'LineWidth', 2, 'color','blue')

legend('$\log_{10}| u(x,t_1)-\check{u}(x,t_1)|$','Interpreter','latex') 
legend('Location','northeast')
set(0,'defaultfigurecolor','w')
xlim([-1, 6])
xlabel('$x$','Interpreter','latex')
ylabel('pointwise error in log10-scale','Interpreter','latex')
set(gca,'FontSize',18);


ax = gcf;
exportgraphics(ax, 'Figures/PtErr_t1.pdf')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%------------------- network solution at t2 -------------------%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% slice of network solution in 2d
load(strcat(path, '/u_pred_t2'))
Z = u_pred_t2;

[X, Y] = meshgrid(-1:0.007:6, 0:0.001:1);

Z = reshape(Z, 1001, 1001);

figure
surf(X, Y, Z)
shading interp
alpha 0.5
set(0, 'defaultfigurecolor', 'w')
xlim([-1, 6])
ylim([0, 1])
pos = axis;
xlabel('$x$','Interpreter','latex', 'Position', [(pos(2)+pos(1))/2, pos(3)-0.15])
ylabel('$\varphi$','Interpreter','latex', 'Position', [pos(1)-0.65, (pos(3) + pos(4))/2])  
zlabel('$\hat{u}(x,t_2,\varphi)$','Interpreter','latex');
set(gca,'FontSize',18);
hold on

X_line = -1:0.007:6;
T_line = 2;
Y_line = 0;
Y_line = Y_line + heaviside(X_line - T_line);

% plot function line
load(strcat(path, '/u_NN_t2'))
u_NN = reshape(u_NN_t2, size(X_line));
M = max(u_NN);
index = find(u_NN >= M - 0.01);
u_NN(min(index)) = nan;
u_NN(max(index)) = nan;
h = plot3(X_line, Y_line, u_NN, 'k', 'LineWidth', 3);
legend(h, {'$\hat{u}(x,t_2,\varphi(x,t_2))$'},'Interpreter','latex','Location','northoutside') 
colorbar

view([-21, 76])

ax = gcf;
exportgraphics(ax, 'Figures/Lifted_uNN_t2.pdf')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load(strcat(path, '/u_NN_t2'))
load(strcat(path, '/u_exact_t2'))
u_Exact = u_Exact_t2;
u_NN2 = u_NN_t2;
x = -1 : 0.007 : 6;

figure
plot(x, u_Exact, 'LineWidth', 2, 'color','red')
hold on
xlim([-1, 6])
plot(x, u_NN2, '--k', 'LineWidth', 2)
legend({'$u(x,t_2)$','$\check{u}(x,t_2)$'},'Interpreter','latex') 

set(0,'defaultfigurecolor','w')
xlim([-1, 6])
xlabel('$x$','Interpreter','latex')
ylim([-0.2 1.2])
ylabel('solution value','Interpreter','latex')
set(gca,'FontSize',18);

ax = gcf;
exportgraphics(ax, 'Figures/Projected_uNN_t2.pdf')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure
load(strcat(path, '/pterr_t2'))
u_err = pterr_t2;
plot(x, log10(abs(u_err)), 'LineWidth', 2, 'color','blue')

legend('$\log_{10}| u(x,t_2)-\check{u}(x,t_2)|$','Interpreter','latex') 
legend('Location','northeast')
set(0,'defaultfigurecolor','w')
xlim([-1, 6])
xlabel('$x$','Interpreter','latex')
ylabel('pointwise error in log10-scale','Interpreter','latex')
set(gca,'FontSize',18);


ax = gcf;
exportgraphics(ax, 'Figures/PtErr_t2.pdf')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%------------------- network solution at t3 -------------------%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% slice of network solution in 2d
load(strcat(path, '/u_pred_t3'))
Z = u_pred_t3;

[X, Y] = meshgrid(-1:0.007:6, 0:0.001:1);

Z = reshape(Z, 1001, 1001);

figure
surf(X, Y, Z)
shading interp
alpha 0.5
set(0, 'defaultfigurecolor', 'w')
xlim([-1, 6])
ylim([0, 1])
pos = axis;
xlabel('$x$','Interpreter','latex', 'Position', [(pos(2)+pos(1))/2, pos(3)-0.15])
ylabel('$\varphi$','Interpreter','latex', 'Position', [pos(1)-0.65, (pos(3) + pos(4))/2])  
zlabel('$\hat{u}(x,t_3,\varphi)$','Interpreter','latex');
set(gca,'FontSize',18);
hold on

X_line = -1:0.007:6;
T_line = sqrt(8);
Y_line = 0;
Y_line = Y_line + heaviside(X_line - T_line);

% plot function line
load(strcat(path, '/u_NN_t3'))
u_NN = reshape(u_NN_t3, size(X_line));
M = max(u_NN);
index = find(u_NN >= M - 0.01);
u_NN(min(index)) = nan;
u_NN(max(index)) = nan;
h = plot3(X_line, Y_line, u_NN, 'k', 'LineWidth', 3);
legend(h, {'$\hat{u}(x,t_3,\varphi(x,t_3))$'},'Interpreter','latex','Location','northoutside') 
colorbar

view([-21, 76])

ax = gcf;
exportgraphics(ax, 'Figures/Lifted_uNN_t3.pdf')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load(strcat(path, '/u_NN_t3'))
load(strcat(path, '/u_exact_t3'))
u_Exact = u_Exact_t3;
u_NN1 = u_NN_t3;
x = -1 : 0.007 : 6;

figure
plot(x, u_Exact, 'LineWidth', 2, 'color','red')
hold on
xlim([-1, 6])
plot(x, u_NN1, '--k', 'LineWidth', 2)
legend({'$u(x,t_3)$','$\check{u}(x,t_3)$'},'Interpreter','latex') 

set(0,'defaultfigurecolor','w')
xlim([-1, 6])
xlabel('$x$','Interpreter','latex')
ylim([-0.2 1.2])
ylabel('solution value','Interpreter','latex')
set(gca,'FontSize',18);

ax = gcf;
exportgraphics(ax, 'Figures/Projected_uNN_t3.pdf')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure
load(strcat(path, '/pterr_t3'))
u_err = pterr_t3;
plot(x, log10(abs(u_err)), 'LineWidth', 2, 'color','blue')

legend('$\log_{10}| u(x,t_3)-\check{u}(x,t_3)|$','Interpreter','latex') 
legend('Location','northeast')
set(0,'defaultfigurecolor','w')
xlim([-1, 6])
xlabel('$x$','Interpreter','latex')
ylabel('pointwise error in log10-scale','Interpreter','latex')
set(gca,'FontSize',18);


ax = gcf;
exportgraphics(ax, 'Figures/PtErr_t3.pdf')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
