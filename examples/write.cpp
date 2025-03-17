

ImageParam input(type_of<uint8_t>(), 3, "image");

Var x, y, c;

Param<int> o1("o1", 0, 0, 10);
Param<int> o2("o2", 0, 0, 10);
Param<int> o3("o3", 0.0, 0, 10);
Param<int> o4("o4", 0, 0, 10);

Func brighter("brighter");
brighter(x, y, c) = clamp(input(x, y, c) + o1 + o2 + o3 + o4, 0, 255);
result(x, y, c) = cast<uint8_t>(brighter(x, y, c));

result.parallel(y, 2).vectorize(x, 4);
