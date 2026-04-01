clear all
close all

Data=[];it=1;
for ord=1:2
for k=3:3:9
   
maxmesh=0.4/k;
%% Step 2: Create the PDE Model
model = createpde(2);

R = [3;  % 3 = Rectangle
          4;  % Number of vertices
          0; 1; 1; 0;  % X-coordinates
          0; 0; 2; 2]; % Y-coordinates

C1 = [3;
          4; 
          0.2; 0.4; 0.4; 0.2; 
          0.5; 0.5; 1; 1]; 
C2 = [3;
          4;  
          0.6; 0.8; 0.8; 0.6; 
          0.5; 0.5; 1; 1];

gm = [R,C1,C2];
ns = char('R','C1','C2')';
sf = 'R-(C1+C2)';

g = decsg(gm,sf,ns);

% Assign the geometry to the PDE model
geometryFromEdges(model, g);

if it==1
figure;
pdegplot(model,EdgeLabels="on")
it=0;
end
%% Step 3: Generate Mesh

if ord==1
mesh = generateMesh(model, 'Hmax', maxmesh, GeometricOrder="linear"); 
elseif ord==2
mesh = generateMesh(model,'Hmax', maxmesh, GeometricOrder="quadratic");
end

%% Step 4: Define PDE Coefficients
syms u(x,y) I(x,y)
system=[-laplacian(u,[x,y])-8*I+5*u;-1.5*laplacian(I,[x,y])+divergence(I*gradient(u,[x y]),[x y])-u./(1+u)+I];
coeffs = pdeCoefficients(system,[u I]);


% Set homog. Neumann BC on outer boundary
applyBoundaryCondition(model,"neumann", ...
                       Edge=[1,2,3,4],g=[0;0]);
% Set Robin BC on inner boundaries
robint=@(location,state)[ones(1,numel(location.x));
        zeros(1,numel(location.x));
        zeros(1,numel(location.x));
        ones(1,numel(location.x))];

applyBoundaryCondition(model,"neumann", ...
                   Edge=[5:12],q=robint,g=[0;0]);

u0 = [1;1];
setInitialConditions(model,u0);

specifyCoefficients(model,'m',0,'d',0,'c',coeffs.c,'a',coeffs.a,'f',[0;0]);

%% Step 6: Solve the PDE
t_start=tic();
result = solvepde(model);
t_end=toc(t_start);
u_numerical = result.NodalSolution;
p = model.Mesh.Nodes; % Get node positions
x_num = p(1,:)';
y_num = p(2,:)';
state.u=u_numerical';
FEM = assembleFEMatrices(model,state);
A = FEM.A;
Data = [Data; [ord, k, length(mesh.Nodes), condest(A), t_end]];

drawQuad=2;
if drawQuad==2
%% Step 8: Visualize the Numerical Solution
if k==3 || k==9
if ord==drawQuad
figure;
pdemesh(model);
title('Finite Element Mesh');
xlabel('x');
ylabel('y');

figure;
pdeplot(model, 'XYData', u_numerical(:,1), 'ZData', u_numerical(:,1),'Contour', 'on', 'Mesh', 'on');
colormap jet;
colorbar;
title('Numerical Solution u(x,y)');
xlabel('x'); ylabel('y');
figure;
pdeplot(model, 'XYData', u_numerical(:,2), 'ZData', u_numerical(:,2),'Contour', 'on', 'Mesh', 'on');
colormap jet;
colorbar;
title('Numerical Solution I(x,y)');
xlabel('x'); ylabel('y');
end
end
end
end
end
t1 = table(Data(:,1),Data(:,2),Data(:,3),Data(:,4),Data(:,5),'VariableNames', {'lin/quad','k', 'Num_nodes', 'Cond_num', 'Time'});
disp(t1);