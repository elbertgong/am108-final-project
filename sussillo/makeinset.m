function hout=makeinset( xy )
% hout=makeinset( xy ) Creates inset in plot
%
% Accepts two click to mark source
% xy (optional) - instead of ginput
% produces a copy of source
%
% Written by Omri Barak

% get source coordinates
if nargin < 1
    xy = ginput(2);
end

% copy plot
a = gca;
h = copyobj(a,gcf);

% position in middle of old plot
p = get(h,'position');
p( [1 2] ) = p([1 2]) + p([3 4])/4;
p([3 4])=p([3 4])/2;
set(h,'position',p);

% set correct zoom
axes(h);
axis( [ min(xy(:,1)) max(xy(:,1)) min(xy(:,2)) max(xy(:,2)) ] );

% bring forward
c = get(gcf,'children');
c( find(c==h) ) = [];
i = find(c==a);
c = [ h; c ];
set(gcf,'children',c);

% return handle
if nargout > 0
    hout = h;
end
