% demo of 'circular PCA' (NLPCA.cir) 
% using the inverse network architecture with circular units
% 
% data: noisy circle
%
% circular units provide components as closed curves 
%
% see also: Scholz et al. BIRD conference, Berlin 2007
%           www.nlpca.org

% Author: Matthias Scholz
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% create circular data

    data = hdf5read('states.hdf5','states1');


% start component extraction

  [pc,net,network]=nlpca(data);
   pc = pc';
   

  
   plot(pc(:,1))
   hold on
   plot(pc(:,2))
   hold on
   plot(pc(:,3))
   %% plot components             
   nlpca_plot(net) 

% save result

  % save nlpca_result_circle   net network

    
%  plot component
    title('{\bf Circular PCA}')
    axis equal
    