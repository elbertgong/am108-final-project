% [f g H] = find_one(x,N,W)
% Auxiliary function to find fixed points of a network
% The network should be described by the following equation:
%   dx/dt = -x + W*tanh(x)
% where W is an N*N matrix.
% Usage:
%   [fixed,fval] = fminunc( @(x) find_one(x, N, W), xstart, ...
%                   optimset('tolfun',1e-10,'hessian','on', ...
%                   'gradobj','on','display','off') );
%   Wp = W.*(ones(N,1)*(1-tanh(fixed).^2)')-eye(N);
%   [vv dd] = eig(Wp);
%
% Return values:
% f - function value
% g - gradient
% H - Hessian (actually Gauss-Newton approximation)
%
% Sussillo D, Barak O.
% Opening the Black Box: Low-dimensional dynamics in high-dimensional
% recurrent neural networks. Neural Computation. 25(3):626-649 (2013)


function [f, g, H] = find_one(x,N,W)
r=tanh(x);
d1 = 1-r.^2;
% d2 = -2*r.*d1;

dx = -x+W*r;
f = 0.5*(dx'*dx);

h = ((W' .* (d1*ones(1,N)) )-eye(N));
g = h*dx;
H = h*h';
% H = h*h' + diag( d2.*(W'*dx));    % This is the full Hessian
end
