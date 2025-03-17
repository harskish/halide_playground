// Copyright 2015 Adobe Systems Incorporated
// All Rights Reserved.

ImageParam input(type_of<uint8_t>(), 3, "image 1");

Var x, y, c;

Func ip;
ip(x, y, c) = cast<float>(BoundaryConditions::repeat_edge(input)(x, y, c));

Param<int> radius("radius", 5, 0, 15);

RDom r(-radius, max(1, 2*radius), -radius, max(1, 2*radius));

// Symolic expression, has no concrete shape yet
Func weight;
weight(x, y) = exp(-((x * x + y * y) / cast<float>(max(1, 2 * radius * radius))));

// No concrete shape yet
Func rweight;
rweight(x, y) = weight(x, y) / sum(weight(r.x, r.y));

// Result has concrete shape
// Kernel rdom has concrete shape => rweight gets shape
result(x, y, c) = cast<uint8_t>(sum(ip(x + r.x, y + r.y, c) * rweight(r.x, r.y)));

//result(x, y, c) = select(x == 1 & y == 1 & c == 1, 255, result(x, y, c));

rweight.compute_root();
result.parallel(y, 1).vectorize(x, 16);

return result;