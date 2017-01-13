load('3d-data.mat');
k=7;
[value,center] = k_means(k,r);
hold on
xdata = r(:,1);
ydata = r(:,2);
zdata = r(:,3);
c = ['r+';'g+';'b+';'m+';'y+';'k+';'c+'];
for i=1:1:k
    location = find(value==i);
    x = xdata(location);
    y = ydata(location);
    z = zdata(location);
    
     plot3(x,y,z,c(i,:));
end
plot3(center(1,:),center(2,:),center(3,:),'o');
hold off