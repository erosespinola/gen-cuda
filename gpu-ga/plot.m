data = csvread('benchmark20tsp.csv');

x = [1 : 10];
y = 100 .* [1 : 10];

z1 = data(:, 3);
z2 = data(:, 4);

[tx, ty] = meshgrid(x, y);
z1 = reshape(z1, size(tx)(1), size(ty)(2));
z2 = reshape(z2, size(tx)(1), size(ty)(2));

surf (tx, ty, z1);
colormap(blue)
set(h, 'cdata',zeros(size(tx)(1)))
hold on;
surf (tx, ty, z2);
colormap(red)
set(h, 'cdata',zeros(size(ty)(2)))

title ('Run time for rectangular 20-city TSP');
xlabel('Population size');
ylabel('Number of generations')
zlabel('Time')
xt = get(gca, 'XTick');
set (gca, 'XTickLabel', 2.^xt);
