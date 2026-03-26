clear all
close all
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
figure;
pdegplot(model,EdgeLabels="on")
%% Step 3: Generate Mesh
maxmesh=0.125;
generateMesh(model, 'Hmax', maxmesh, GeometricOrder="linear");
figure;
pdemesh(model);
title('Finite Element Mesh');
xlabel('x');
ylabel('y');


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
        1-state.u(1,:)];
applyBoundaryCondition(model,"neumann", ...
                   Edge=[5:12],q=robint,g=[0;0]);

u0 = [1;1];
setInitialConditions(model,u0);

specifyCoefficients(model,'m',0,'d',[1;1],'c',coeffs.c,'a',coeffs.a,'f',[0;0]);

initfun = @(loc)[exp(-10*(loc.x-1/2).^2-10*(loc.y-3/2).^2);
    exp(-15*(loc.x-1/2).^2-15*(loc.y-1/4).^2)];
setInitialConditions(model,initfun);


tlist = 0:0.2:1;
results = solvepde(model,tlist);

u_numerical = results.NodalSolution;


[nNodes, nCols] = size(u_numerical);

if nCols == 2
    sol1 = u_numerical; 
    mag = sqrt(u_numerical(:,1).^2 + u_numerical(:,2).^2);
    times = 0;
elseif mod(nCols,2)==0
    nTimes = nCols/2;
    sol1 = zeros(nNodes, 2*nTimes);
    mag = zeros(nNodes, nTimes);
    for k=1:nTimes
        comp1 = u_numerical(:,2*(k-1)+1);
        comp2 = u_numerical(:,2*(k-1)+2);
        sol1(:,2*(k-1)+1) = comp1;
        sol1(:,2*(k-1)+2) = comp2;
        mag1(:,k) =comp1;
        mag2(:,k) =comp2;
    end
    times = 1:nTimes;
else
    [nRows, nTimes] = size(u_numerical);
    if nRows == 2*nNodes
        mag = zeros(nNodes, nTimes);
        sol1 = zeros(nNodes, 2*nTimes);
        for k=1:nTimes
            comp1 = u_numerical(1:nNodes,k);
            comp2 = u_numerical(nNodes+1:end,k);
            sol1(:,2*(k-1)+1) = comp1;
            sol1(:,2*(k-1)+2) = comp2;
            mag1(:,k) =comp1;
            mag2(:,k) =comp2;

        end
        times = 1:nTimes;
    else
        error('Unexpected shape of results.NodalSolution. Dimensions: %d x %d', nRows, nCols);
    end
end

idxs = [1,2,3,4];
figure;
nplot = numel(idxs);
for i=1:nplot
    subplot(2,2,i)
    pdeplot(model, 'XYData', mag1(:,idxs(i)), 'ZData', mag1(:,idxs(i)), 'Mesh','on');
    title(sprintf('u at t=%f', (idxs(i)-1)*0.2));
    colormap jet; colorbar;
end
figure;
nplot = numel(idxs);
for i=1:nplot
    subplot(2,2,i)
    pdeplot(model, 'XYData', mag2(:,idxs(i)), 'ZData', mag2(:,idxs(i)), 'Mesh','on');
    title(sprintf('I at t=%f', (idxs(i)-1)*0.2));
    colormap jet; colorbar;
end
