load('2d-data.mat');
k=5;
[value,center] = k_means(k,r);
hold on
xdata = r(:,1);
ydata = r(:,2);
z = ['r+';'g+';'b+';'m+';'y+'];
for i=1:1:k
    location = find(value==i);
    x = xdata(location);
    y = ydata(location);
     plot(x,y,z(i,:));
end
plot(center(1,:),center(2,:),'o');
hold off