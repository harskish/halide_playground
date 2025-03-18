ImageParam cfa(type_of<uint16_t>(), 3, "cfa"); // raw fca image
ImageParam colors(type_of<uint8_t>(), 3, "colors"); // color idx per pixel (order: RGBG)
ImageParam pattern(type_of<uint8_t>(), 3, "pattern"); // smallest repeatable bayer pattern, 6x6 for xtrans

Param<int> white_level("white_level", 1023, 0, 65535);
Param<int> black_level("black_level", 1023, 0, 65535);
Param<float> wb_r("camera_whitebalance_r", 1, 0, 1024);
Param<float> wb_g("camera_whitebalance_g", 1, 0, 1024);
Param<float> wb_b("camera_whitebalance_b", 1, 0, 1024);

// https://halide-lang.org/docs/namespace_halide.html#a55158f5f229510194c425dfae256d530

Var x, y, c;

Func ip, ip_colors;
ip(x, y, c) = cast<float>(BoundaryConditions::repeat_edge(cfa)(x, y, c));
ip_colors = BoundaryConditions::repeat_edge(colors);

Func in_range;
Expr col_range = cast<float>(white_level - black_level);
in_range(x, y, c) = max(0, ip(x, y, c) - black_level) / col_range;

Expr wb_min = min(wb_r, min(wb_g, wb_b));
Func wb;
wb(c) = cast<float>(select(c == 0, wb_r, select(c == 1, wb_g, wb_b)) / wb_min);

Func masked;
masked(x, y, c) = select(ip_colors(x, y, 0) == c, in_range(x, y, 0) * wb(c), 0);
//masked(x, y, c) = select(colors(x, y, 0) == c, in_range(x, y, 0), 0) * wb_min;

result(x, y, c) = cast<uint16_t>(clamp(col_range * masked(x, y, c), 0, 3048));
//result(x, y, c) = cast<uint16_t>(clamp(col_range * masked(x, y, c), 0, 65535)); // output: RGB

//wb.compute_root();

return result;