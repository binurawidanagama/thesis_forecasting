% =====================================================================
% src/asp/energy_physics.lp
%
% Thesis-grade ASP “repair selector” for ENERGY forecasting outputs:
%   targets: wind, solar, load
%
% Your Python should provide these facts (all integers, consistently scaled):
%   sample(S).                     % sample id (0..BATCH-1 or global)
%   horizon(H).                    % horizon step (1..Hz)
%   pred(T,S,H,V).                 % predicted value V for target T at (S,H)
%   night(S,H).                    % true if (S,H) is night-time (for solar)
%   month(S,M).                    % month M in 1..12 (based on timestamp for sample S)
%   cap(wind,S,Cw). cap(solar,S,Cs).   % installed capacity (optional but recommended)
%   load_cap(S,Cl).                % optional upper bound for load (if you have it)
%
% Output (what Python parses):
%   repair(Kind, Target, Sample, Horizon).
%
% Kinds:
%   bound_low     -> clamp to >= 0
%   bound_cap     -> clamp to capacity (wind/solar; optional load_cap)
%   pv_night      -> force solar to 0 at night
%   seasonal_cap  -> clamp solar to monthly CF ceiling (based on capacity)
% =====================================================================

% ---------------------------
% Targets we reason about
% ---------------------------
target(wind; solar; load).

% ---------------------------
% Decision: keep OR choose exactly one repair-kind
% (Grounding stays small because you batch in Python.)
% ---------------------------
1 {
    keep(T,S,H);
    repair(bound_low,    T,S,H);
    repair(bound_cap,    T,S,H);
    repair(pv_night,     T,S,H);
    repair(seasonal_cap, T,S,H)
} 1 :- target(T), pred(T,S,H,_), sample(S), horizon(H).

% ---------------------------
% Violation detectors
% ---------------------------

% 1) Non-negativity for all
viol_low(T,S,H) :- target(T), pred(T,S,H,V), V < 0.

% 2) Capacity ceilings (only if cap facts exist)
viol_cap(wind,S,H)  :- pred(wind,S,H,V),  cap(wind,S,C),  V > C.
viol_cap(solar,S,H) :- pred(solar,S,H,V), cap(solar,S,C), V > C.

% Optional: if you provide a load_cap(S,C) fact from Python
viol_cap(load,S,H)  :- pred(load,S,H,V),  load_cap(S,C),  V > C.

% 3) Solar at night must be zero (we trigger if !=0 so it fixes negatives too)
viol_pv_night(S,H) :- night(S,H), pred(solar,S,H,V), V != 0.

% 4) Seasonal solar ceiling (monthly max capacity factor)
%    Only applies if month/2 and cap(solar,...) exist.
max_solar_cf(1,  40).  % Jan
max_solar_cf(2,  55).  % Feb
max_solar_cf(3,  80).  % Mar
max_solar_cf(4,  95).  % Apr
max_solar_cf(5,  100). % May
max_solar_cf(6,  100). % Jun
max_solar_cf(7,  100). % Jul
max_solar_cf(8,  95).  % Aug
max_solar_cf(9,  85).  % Sep
max_solar_cf(10, 65).  % Oct
max_solar_cf(11, 45).  % Nov
max_solar_cf(12, 35).  % Dec

viol_seasonal(S,H) :-
    pred(solar,S,H,V),
    cap(solar,S,C),
    month(S,M),
    max_solar_cf(M,LimitPct),
    Allowed = (C * LimitPct) / 100,
    V > Allowed.

% Aggregate: “any violation exists”
viol_any(T,S,H) :- viol_low(T,S,H).
viol_any(wind,S,H)  :- viol_cap(wind,S,H).
viol_any(solar,S,H) :- viol_cap(solar,S,H).
viol_any(load,S,H)  :- viol_cap(load,S,H).
viol_any(solar,S,H) :- viol_pv_night(S,H).
viol_any(solar,S,H) :- viol_seasonal(S,H).

% ---------------------------
% Keep is forbidden if any rule is violated
% ---------------------------
:- keep(T,S,H), viol_any(T,S,H).

% ---------------------------
% Repair admissibility (robustness):
% Only allow repairs that actually make sense for the detected violation(s).
% This avoids “random repairs” and keeps solutions stable.
% ---------------------------

% bound_low is valid for any negative prediction
ok(bound_low, T,S,H) :- viol_low(T,S,H).

% bound_cap is valid only when a cap-type rule is violated
ok(bound_cap, wind,S,H)  :- viol_cap(wind,S,H).
ok(bound_cap, solar,S,H) :- viol_cap(solar,S,H).
ok(bound_cap, load,S,H)  :- viol_cap(load,S,H).

% pv_night is valid only at night when solar != 0
ok(pv_night, solar,S,H) :- viol_pv_night(S,H).

% seasonal_cap only if seasonal ceiling violated
ok(seasonal_cap, solar,S,H) :- viol_seasonal(S,H).

% If a repair is chosen, it must be admissible.
:- repair(K,T,S,H), not ok(K,T,S,H).

% ---------------------------
% Optimization (efficiency / minimal intervention)
% Prefer the least “invasive” repair that fixes the issue.
% (pv_night and seasonal_cap are “physics/astronomy” fixes, cheaper than hard clamps)
% ---------------------------

:~ repair(bound_low,    T,S,H). [10@1,T,S,H]
:~ repair(bound_cap,    T,S,H). [10@1,T,S,H]
:~ repair(pv_night,     T,S,H). [ 5@1,T,S,H]
:~ repair(seasonal_cap, T,S,H). [ 5@1,T,S,H]

#show repair/4.
