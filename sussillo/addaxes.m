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

