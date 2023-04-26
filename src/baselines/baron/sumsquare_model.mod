set I := 1 .. 10;
var x{I} <= 10, >= -10;

minimize SSy: sum{i in I} (i * x[i]);
subject to ctrs{i in I}: 1 <= x[i] <= 10;