// Copyright 2015 Adobe Systems Incorporated
// All Rights Reserved.

ImageParam input(type_of<uint8_t>(), 3, "image 1");

Var x, y, c;

Func ip;
ip(x, y, c) = cast<float>(BoundaryConditions::repeat_edge(input)(x, y, c));

Param<int> radius("radius", 5, 1, 35);

RDom r(-(radius/2), radius, -(radius/2), radius);
//RDom r(-(radius/2), radius, 0, 1);
//RDom r(0, 1, -(radius/2), radius);

// Symolic expression, has no concrete shape yet
Func weight;
weight(x, y) = exp(-((x * x + y * y) / cast<float>(2 * radius * radius)));

// No concrete shape yet
Func rweight;
rweight(x, y) = weight(x, y) / sum(weight(r.x, r.y));

// Result has concrete shape
// Kernel rdom has concrete shape => rweight gets shape
result(x, y, c) = cast<uint8_t>(sum(ip(x + r.x, y + r.y, c) * rweight(r.x, r.y)));

//ip(x, y, c) = select(x == 0 & y == 0 & c == 0, 255, ip(x, y, c));

rweight.compute_root();
result.parallel(y, 2).vectorize(x, 8);

return result;
