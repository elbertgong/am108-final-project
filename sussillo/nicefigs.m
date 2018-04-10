function h = nicefigs( operation, varargin)
% Create figures for papers
%
% fig = nicefigs( 'create', [ W H ] );
%    create fig of W x H centimeters
% fig = nicefigs( 'create', [ W H ], 0 );
%    Figure size is full paper instead of tight
% pan = nicefigs( 'add', [ W H ], newAnchor, oldObj, oldAnchor, [ X Y ] );
%    Add axes sized W*H. newAnchor will be X,Y relative to oldAnchor of oldObj
% nicefigs( 'label', ax, txt, <size>, <ofst> )
%    Adds a label on the top left of an axis
%
%
% Example:
%   fig = nicefigs( 'create', [13 4] );
%   pan(1) = nicefigs( 'add', [4 2],'left-top',fig,'left-top',[1.5 -1] );
%   pan(2) = nicefigs( 'add', [4 2],'left-top',pan(1),'right-top',[2 0] );
%
%
%
% Written by Barak Blumenfeld
% Modified by Omri Barak

switch lower(operation)
    case 'create'
        h = figure;
        set(h,'units','centimeters','color',[1 1 1]);
        set(h,'position',[ 1 1 varargin{1}]);
        set(h,'paperunits',get(h,'units'));
        temp2=get(h,'position');
        temp3=temp2(3:4);
        tightFit = 1; % Adjust paper size to figure
        if nargin>2
            tightFit = varargin{2};
        end
        if tightFit
            set(h,'papersize',temp3*1.05);
        end
        temp=get(h,'paperSize');
        margs= (temp-temp3)/2;
        set(h,'paperposition',[margs temp2(3:4)])
    case 'add'
        h = addaxes( varargin{:} );
case 'label'
    pan_text = addaxes([1 1]*0.7,'right-top',varargin{1},'left-top',[-.3 0.7]);
    axes(pan_text);
    text(0,0.5, varargin{2},'fontsize',11,'FontWeight','bold','fontname','timesNewRoman');
    axis off;
end


% Subfunctions
%---------------
    function h=addaxes(axsiz,axref,obj,objref,ofst)
        % Written by Barak Blumenfeld
        if ~exist('ofst')
            ofst=[0 0];
        end
        
        
        %units
        temp=get(obj,'units');
        if ~isa(temp,'cell')
            temp={temp};
        end
        if all(strcmp(temp,temp{1}))
            objunits=temp{1};
        else
            error('all objects must have same units');
        end
        
        
        
        objposall = get(obj,'position');
        %position
        if ~isa(objposall,'cell')
            objposall={objposall};
        end
        
        temp= get(obj,'type');
        if ~isa(temp,'cell')
            temp={temp};
        end
        temp2=find(strcmp(temp,'figure'));
        if length(temp2)>1
            error('only one figure supported');
        elseif length(temp2)==1
            objpos=objposall{temp2};
            objpos(1:2)=[0 0];
        else
            objpos=objposall{1};
            for i=2:length(obj)
                temp3=objposall{i};
                objpos(1)=min(objpos(1),temp3(1));
                objpos(2)=min(objpos(2),temp3(2));
                temp4=max(objpos(1)+objpos(3),temp3(1)+temp3(3));
                objpos(3)=temp4-objpos(1);
                temp5=max(objpos(2)+objpos(4),temp3(2)+temp3(4));
                objpos(4)=temp5-objpos(2);
            end
        end
        
        %    objpos(1:2)=[0 0];
        %objtype = get(obj,'type');
        
        % =get(obj,'units');
        %if isequal(objtype,'figure');
        %   objpos(1:2)=[0 0];
        %end
        objp   = getoffset(objpos(3:4),objref);
        axp    = getoffset(axsiz,axref);
        axpos = [objpos(1:2)+objp-axp+ofst  axsiz] ;
        h=axes('units',objunits,'position',axpos);
        
    end

    function ofst=getoffset(bx,loc)
        h=[];
        if length(bx)==1 %handle
            h=bx;
            pos=get(h,'position');
            siz=pos(3:4);
        elseif length(bx)==2
            siz = bx;
        elseif length(bx)==4
            siz = bx(3:4)
        else
            error('unknown box argument');
        end
        
        cntr = siz/2;
        
        switch loc
            case {'top-left','left-top'}
                ofst = [0       siz(2) ];
            case {'top-right','right-top'}
                ofst = [siz(1)  siz(2) ];
            case {'bottom-left','left-bottom'}
                ofst = [0       0      ];
            case {'bottom-right','right-bottom'}
                ofst = [siz(1)  0      ];
            case {'top-center','center-top'}
                ofst = [cntr(1) siz(2) ];
            case {'bottom-center','center-bottom'}
                ofst = [cntr(1) 0      ];
            case {'left-center','center-left'}
                ofst = [0       cntr(2)];
            case {'right-center','center-right'}
                ofst = [siz(1)  cntr(2)];
            case 'center'
                ofst = [cntr(1)  cntr(2)];
            otherwise
                error ('unknown location type');
        end
    end


end