% src/asp/core_asp.lp

target(wind; solar; load; price).

% default choice (we may later expand this)
{ keep(T,S,H) } :- target(T), sample(S), horizon(H), pred(T,S,H,_).

% for non-price targets, exactly one of keep / repairs
1 { keep(T,S,H); repair(bound_low,T,S,H); repair(bound_cap,T,S,H); repair(pv_night,T,S,H) } 1
    :- T != price, sample(S), horizon(H), pred(T,S,H,_).

% --- non-negativity in scaled units ---
:- keep(T,S,H), pred(T,S,H,V), V < 0, T != price.
:- pred(T,S,H,V), V < 0, T != price, not repair(bound_low,T,S,H).

% --- capacity bounds (wind/solar) ---
% cap/3 is also scaled by 'scale'
:- keep(wind,S,H),  pred(wind,S,H,V),  cap(wind,S,C),  V > C.
:- keep(solar,S,H), pred(solar,S,H,V), cap(solar,S,C), V > C.

% --- PV night rule ---
% e.g. 1.0 MW => 1.0 * scale
night_threshold(100).   % scaled: 1.0 MW * 100

:- keep(solar,S,H),
   night(S,H),
   pred(solar,S,H,V),
   night_threshold(T), V > T.

% --- ramp rules (in scaled units) ---
% e.g. 800 MW/h -> 800*scale etc.
ramp_limit(wind,  80000).
ramp_limit(solar, 50000).
ramp_limit(load,  150000).
ramp_limit(price, 25000).

val(T,S,H,V) :- pred(T,S,H,V).

big_ramp_up(T,S,H) :-
    val(T,S,H,V2), val(T,S,H-1,V1),
    ramp_limit(T,R), V2 - V1 > R, H > 1, T != price.

big_ramp_down(T,S,H) :-
    val(T,S,H,V2), val(T,S,H-1,V1),
    ramp_limit(T,R), V1 - V2 > R, H > 1, T != price.

% --- prefer minimal repairs ---
:~ repair(bound_low,T,S,H). [1@1,T,S,H]
:~ repair(bound_cap,T,S,H). [1@1,T,S,H]
:~ repair(pv_night,T,S,H).  [1@1,T,S,H]

#show repair/4.
