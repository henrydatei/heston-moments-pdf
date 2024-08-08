(* Define the variables *)
z = Symbol["z"];
k = Symbol["k"];

(* Define the equation *)
eq = (72 - 72*z^2)/(z^6 - 3*z^4 + 9*z^2 + 9) == k;

(* Solve the equation *)
solutions = Solve[eq, z];

(* Display results *)
Print[solutions];
