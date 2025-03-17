
ImageParam input(type_of<uint16_t>(), 3, "cfa_in");

// Assumes fuji x-trans format

Var x, y, c;

Func ip;
ip(x, y, c) = cast<float>(BoundaryConditions::repeat_edge(input)(x, y, c));

//Func normalized;
//normalized(x, y, c) = clamp(ip(x, y, c) / 4095.0f, 0, 1);

//Func r, g, b;
//r(x, y, c) = 

result(x, y, c) = cast<uint16_t>(ip(x, y, c));

return result;