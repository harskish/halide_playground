ImageParam input(type_of<uint8_t>(), 3, "image 1");

Var x, y, c;

Func brighter("brighter");
brighter(x, y, c) = clamp(input(x, y, c) + 1, 0, 255);
result(x, y, c) = cast<uint8_t>(brighter(x, y, c));

result.parallel(y, 2).vectorize(x, 4);
