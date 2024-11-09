function plot_solu

format short e
close all
path = 'Result/simulation_0';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('NumberTitle', 'off', 'Name', 'Network Solution in 3D');


for i = 0 : 1 : 1
    [X1, Y1] = meshgrid(-1:0.007:6, 0:0.01:4);
    [X2, Y2] = meshgrid(-1:0.007:6, 4.01:0.01:10);
    X = [X1; X2];
    Y = [Y1; Y2];
    Z = i * ones(size(X));
    g = ['/u_NN_level' num2str(i)];
    load(strcat(path, g));
    C = u_NN_level;
    C = reshape(C, 1001, 1001);
    if i == 0
        mask1 = X1 - 0.25 * Y1 - 1 > 0;
        mask2 = X2 - sqrt(Y2) > 0;
        mask = [mask1; mask2];
    else
        mask1 = X1 - 0.25 * Y1 - 1 <= 0;
        mask2 = X2 - sqrt(Y2) <= 0;
        mask = [mask1; mask2];
    end
    C(mask) = nan;

    surf(X, Y, Z, C)
    c = colorbar;
    set(c,'TickLabelInterpreter','latex')
    shading interp
    alpha 0.5
    set(0,'defaultfigurecolor','w')
    xlim([-1, 6])
    ylim([0, 10])
    pos = axis;
    axis equal
    xlabel('$x$','Interpreter','latex', 'Position', [(pos(2)+pos(1))/2+0.7, pos(3)-0.2])
    ylabel('$t$','Interpreter','latex', 'Position', [pos(1)-1.0, (pos(1) + pos(2))/2 +0.5])  
    zlabel('$\varphi(x,t)$','Interpreter','latex');
    set(gca,'FontSize',18);
    set(gca, 'Zticklabel', {'$0$',' ', '$1$'}, 'TickLabelInterpreter', 'latex');
    hold on
end

ax = gca;
cb = colorbar('Location','eastoutside');
ax.Position = ax.Position - [0 0 .02 .02];
cb.Position = cb.Position + [.02 0 0 0];
clim([0 1])
view(-27, 36)
axis equal

ax = gcf;
exportgraphics(ax, 'Figures/Lifted_u_NN.pdf')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% network solution in 2d
load(strcat(path, '/u_NN_projected'))

[X, Y] = meshgrid(-1:0.007:6, 0:0.01:10);
C_u = reshape(u_pred, 1001, 1001);

figure('NumberTitle', 'off', 'Name', 'Projection of Network Solution in 2D')
surf(X, Y, C_u)
colorbar
shading interp
grid off
alpha 0.5
set(1, 'defaultfigurecolor', 'w')
xlim([-1, 6])
ylim([0, 10])
clim([0, 1])
xlabel('$x$','Interpreter','latex', 'Position', [(pos(1) + pos(2))/2+0.5, pos(3)-0.15])
ylabel('$t$','Interpreter','latex', 'Position', [pos(1)-0.18, (pos(3) + pos(4))/2]) 
set(gca,'FontSize',18);
view(0,90)

axis tight

ax = gcf;
exportgraphics(ax, 'Figures/Projected_uNN.pdf')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% pointwise error in 2d
load(strcat(path, '/pterr'))
u_err = u_error;

figure('NumberTitle', 'off', 'Name', 'Pointwise Error in 2D');
C22 = abs(u_err);
C22 = reshape(C22, 1001, 1001);
surf(X, Y, C22)
colorbar
shading interp
grid off
alpha 0.5
set(1, 'defaultfigurecolor', 'w')
xlim([-1, 1])
ylim([0, 1])
xlabel('$x$','Interpreter','latex', 'Position', [(pos(2)+pos(1))/2+0.5, pos(3)-0.15])
ylabel('$t$','Interpreter','latex', 'Position', [pos(1)-0.18, (pos(3) + pos(4))/2]) ;
set(gca,'FontSize',18);
view(0, 90)

axis tight

ax = gcf;
exportgraphics(ax, 'Figures/Pterr.pdf')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% network solution in 2d
load(strcat(path, '/u_exact'))

[X, Y] = meshgrid(-1:0.007:6, 0:0.01:10);
C_u = reshape(u_exact, 1001, 1001);

figure('NumberTitle', 'off', 'Name', 'Exact solution in 2D')
surf(X, Y, C_u)
colorbar
shading interp
grid off
alpha 0.5
set(1, 'defaultfigurecolor', 'w')
xlim([-1, 6])
ylim([0, 10])
clim([0, 1])
xlabel('$x$','Interpreter','latex', 'Position', [(pos(1) + pos(2))/2+0.5, pos(3)-0.15])
ylabel('$t$','Interpreter','latex', 'Position', [pos(1)-0.18, (pos(3) + pos(4))/2]) 
set(gca,'FontSize',18);
view(0,90)

axis tight

ax = gcf;
exportgraphics(ax, 'Figures/u_exact.pdf')




