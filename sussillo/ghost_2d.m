% Sussillo D, Barak O.
% Opening the Black Box: Low-dimensional dynamics in high-dimensional
% recurrent neural networks. Neural Computation. 25(3):626-649 (2013)
%
% Code to produce Figure 5


clear all

fig = nicefigs( 'create', [16 8] );

sz = [5 5];
pan(1) = addaxes(sz,'left-top',fig,'left-top',[1.5 -1.5]);
pan(2) = addaxes(sz,'left-top',pan(1),'right-top',[2.5 0]);

pan_strings = 'AB';
for i=1:2
    pan_text(i)=addaxes([1 1]*0.7,'right-top',pan(i),'left-top',[-0.7 1.5]);
    axes(pan_text(i));
    text(0,0.5,pan_strings(i),'fontsize',12,'FontWeight','bold');
    axis off;
end

%%
as = [-0.3 0.3];
for iParam = 1:2
    axes( pan(iParam) );
    a = as(iParam);
    xlabel( 'x_1' );
    ylabel('x_2');
    % axis([0.5 1.5 -0.5 0.5]);
    % grid;
    hold on;
    
    
    f1 = @(x,y) y - (x.^2+.25+a);
    f2 = @(x,y) x-y;
    f = @(x,y) [ f1(x,y); f2(x,y) ];
    g = @(x,y) 0.5 * (f1(x,y).^2 + f2(x,y).^2);
    
    x=0:.01:1;
    y=-0:.01:1;
    x=-5:.05:5;
    y=x;
    eta=1e1;
    
    [X Y]=meshgrid(x,y);
    
    % contour(x,y,g(X,Y),150)
    x=-5:.01:5
    plot(x,x,'g','linewidth',2)
    plot(x,x.^2+.25+a,'g','linewidth',2)
    axis([-4 4 -4 4]);
    
    trajIC = [
        
    -1.931    5.1316  30.0000
    -1.7396    5.0731  100.0000
    -1.4862    5.0146  100.0000
    -0.9332    5.0731  100.0000
    0.7258    5.1901  85.0000
    5.2880    3.7573  1000.0000
    5.1728    1.9152  80.0000
    5.1498   -0.4240  70.0000
    5.2880   -2.3246  90.0000
    5.1498   -4.0789  90.0000
    5.1498   -4.9561  100.0000
    3.6751   -5.3363  100.0000
    2.5691   -5.3363  100.0000
    1.3940   -5.3070  70.0000
    ];

for j=1:size(trajIC,1)
    x = trajIC(j,1:2)';
    
    T = 20;
    dt = 1e-3;
    sampEvery = 0.01 / dt;
    X = zeros(2, T*sampEvery);
    k=1;
    for t=0:T/dt
        
        dx = f(x(1),x(2));
        x(1) = x(1) + dt*( dx(1) );
        x(2) = x(2) + dt*( dx(2) );
        clr='k';
        if ~mod(t,sampEvery)
            X(:,k)=x;
            k=k+1;
        end
    end
    plot(X(1,:),X(2,:),clr,'linewidth',2);
    [ax ay] = dsxy2figxy(X(1,trajIC(j,3)+[0 1]),X(2,trajIC(j,3)+[0 1]) );
    har = annotation( 'arrow', ax,ay );
    set(har, 'HeadWidth', 15 );
    %         plot(X(1,end),X(2,end), 'xk','markersize',20,'linewidth',5);
end

[fixed val eFlag]= fminunc( @(z) g(z(1),z(2)),[0 0],optimset('tolfun',1e-8,'hessian','off','gradobj','off','display','off') );
plot(fixed(1),fixed(2),'.r','linewidth',3,'markersize',15)
[fixed val eFlag]= fminunc( @(z) g(z(1),z(2)),[5 5],optimset('tolfun',1e-8,'hessian','off','gradobj','off','display','off') );
plot(fixed(1),fixed(2),'.r','linewidth',3,'markersize',15)
J=[-2*fixed(1) 1;1 -1];
[v d] = eig(J);

end

hInset = makeinset([-0.2 1.4; 1.15 -0.1]);
box on
set(gca,'xtick',[])
set(gca,'ytick',[])
xlabel('')
ylabel('')
set(gca,'position',[ 13.0525    2.3378    2.5000    2.5000] );

a = 1.5;
plot( fixed(1) + a*[-1 -0.1 nan -1 -0.1 nan 0.1 1 nan 0.1 1], fixed(2) + ...
    a*[-1 -0.1 nan 1 0.1 nan 0.1 1 nan -0.1 -1], 'm', 'linewidth',2 )
[ax ay] = dsxy2figxy( fixed(1)+a*[-.2 -.1], fixed(2)+a*[.2 .1] );
har = annotation( 'arrow', ax,ay );
set(har, 'HeadWidth', 15, 'color', 'm' );
[ax ay] = dsxy2figxy( fixed(1)+a*[.2 .1], fixed(2)+a*[-.2 -.1] );
har = annotation( 'arrow', ax,ay );
set(har, 'HeadWidth', 15, 'color', 'm' );

%%
print -dpdf Figs/figure_ghost.pdf
