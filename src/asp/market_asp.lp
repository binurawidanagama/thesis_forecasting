% asp/market_asp.lp

% -100.0 .. 1000.0 -> scaled
price_min(-10000).
price_max(100000).

1 { keep(price,S,H); repair(bound_low,price,S,H) } 1 :-
    sample(S), horizon(H), pred(price,S,H,_).

:- keep(price,S,H),
   pred(price,S,H,V),
   price_min(MN), V < MN.

:- keep(price,S,H),
   pred(price,S,H,V),
   price_max(MX), V > MX.

:~ repair(bound_low,price,S,H). [1@1,price,S,H]

#show repair/4.