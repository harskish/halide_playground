ImageParam input(type_of<uint8_t>(), 3, "image 1");

Var x, y, c;
result(x, y, c) = cast<uint8_t>(input(x, y, c) + 1);

result.parallel(y, 2).vectorize(x, 4);
