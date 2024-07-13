
function clustered = k_means(events, CLUSTERS)
       len = length(events);    % number of events
       cl = len/CLUSTERS;       % step size between centroids
       mu = ones(CLUSTERS, 2);   % array of centroids
       clustered = [events,ones(len,1)];

       % initialize clusters
       for c = 1:CLUSTERS
           ind = floor(cl)*c;
           mu(c,:) = events(ind,:);
       end

       % assign events to cluster centroids
       for i = 1:len
           [centroid,ind] = pdist2(mu,clustered(i,[1 2]),'euclidean','Smallest',1);
           clustered(i, end) = ind
       end

       % move cluster centroids
       for k = 1:CLUSTERS
           s=clustered(:,end) ==k;
           h = mean(clustered(s,[1 2]));
           mu(k,:) = h;  
       end
end