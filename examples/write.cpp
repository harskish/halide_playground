ImageParam input(type_of<uint8_t>(), 3, "image 1");

Var x, y, c;

Param<int> offset("offset", 0, -60, 60);

Func brighter("brighter");
brighter(x, y, c) = clamp(input(x, y, c) + offset, 0, 255);
result(x, y, c) = cast<uint8_t>(brighter(x, y, c));

//result.parallel(y, 2).vectorize(x, 4);

result(x, y, c) = select(x == 1 & y == 1, 255, result(x, y, c));