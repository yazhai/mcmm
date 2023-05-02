\ Model levy
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  var_summation_item0 + var_summation_item1 + [ 2 sin_1_out ^2
   + 2 term_last_1 * term_last_2 ] / 2 
Subject To
 w0: - 0.25 x0 + w0 = 0.75
 w1: - 0.25 x1 + w1 = 0.75
 w2: - 0.25 x2 + w2 = 0.75
 sin_1_in: - 3.141592653589793 w0 + sin_1_in = 0
 sin_last_in: - 6.283185307179586 w2 + sin_last_in = 0
 const_sin_inter_in0: - 3.141592653589793 w0 + var_sin_inter_in0 = 1
 const_sin_inter_in1: - 3.141592653589793 w1 + var_sin_inter_in1 = 1
 x0_lb: x0 >= -10
 x0_ub: x0 <= 10
 x1_lb: x1 >= -10
 x1_ub: x1 <= 10
 x2_lb: x2 >= -10
 x2_ub: x2 <= 10
 term_last_1: 2 w2 + term_last_1 + [ - w2 ^2 ] = 1
 term_last_2: term_last_2 + [ - sin_last_out ^2 ] = 1
 const_term_inter_10: 2 w0 + var_term_inter_10 + [ - w0 ^2 ] = 1
 const_term_inter_20: var_term_inter_20 + [ - 10 var_sin_inter_out0 ^2 ]
   = 1
 const_summation_item0: var_summation_item0 + [
   - var_term_inter_10 * var_term_inter_20 ] = 0
 const_term_inter_11: 2 w1 + var_term_inter_11 + [ - w1 ^2 ] = 1
 const_term_inter_21: var_term_inter_21 + [ - 10 var_sin_inter_out1 ^2 ]
   = 1
 const_summation_item1: var_summation_item1 + [
   - var_term_inter_11 * var_term_inter_21 ] = 0
Bounds
General Constraints
 sin_1_out: sin_1_out = SIN ( sin_1_in )
 sin_last_out: sin_last_out = SIN ( sin_last_in )
 const_sin_inter_out0: var_sin_inter_out0 = SIN ( var_sin_inter_in0 )
 const_sin_inter_out1: var_sin_inter_out1 = SIN ( var_sin_inter_in1 )
End
