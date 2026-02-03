% =====================================================================
% src/asp/energy_physics.lp  (FAST deterministic repairs)
%
% Facts expected (integers scaled by SCALE in Python):
%   sample(S).
%   horizon(H).
%   pred(wind|solar|load, S, H, V).
%   night(S,H).                     % optional
%   month(S,M).                     % optional (1..12)
%   cap(wind,S,C). cap(solar,S,C).  % optional (scaled)
%
% Output (what Python expects):
%   repair(kind, target, sample, horizon).
%
% kind ∈ { bound_low, bound_cap, pv_night, seasonal_cap }
% =====================================================================

target(wind; solar; load).

#show repair/4.

% ---------------------------
% Seasonal solar CF limits (% of capacity)
% ---------------------------
max_solar_cf(1,  40).
max_solar_cf(2,  55).
max_solar_cf(3,  80).
max_solar_cf(4,  95).
max_solar_cf(5,  100).
max_solar_cf(6,  100).
max_solar_cf(7,  100).
max_solar_cf(8,  95).
max_solar_cf(9,  85).
max_solar_cf(10, 65).
max_solar_cf(11, 45).
max_solar_cf(12, 35).

% ---------------------------
% Violations
% ---------------------------

% Non-negativity for all
viol_low(T,S,H) :- target(T), pred(T,S,H,V), V < 0.

% Capacity ceilings (only if cap facts exist)
viol_cap(wind,S,H)  :- pred(wind,S,H,V),  cap(wind,S,C),  V > C.
viol_cap(solar,S,H) :- pred(solar,S,H,V), cap(solar,S,C), V > C.

% Night rule: solar must be 0 at night
viol_night(S,H) :- night(S,H), pred(solar,S,H,V), V != 0.

% Seasonal solar ceiling (avoid division: compare V*100 > C*LimitPct)
viol_seasonal(S,H) :-
    pred(solar,S,H,V),
    cap(solar,S,C),
    month(S,M),
    max_solar_cf(M,LimitPct),
    not night(S,H),
    V * 100 > C * LimitPct.

% ---------------------------
% Repairs with precedence (one repair per cell)
% Precedence:
%   pv_night > seasonal_cap > bound_cap > bound_low
% ---------------------------

% 1) Night solar dominates
repair(pv_night, solar, S,H) :- viol_night(S,H).

% 2) Seasonal clamp for solar (only if not night)
repair(seasonal_cap, solar, S,H) :-
    viol_seasonal(S,H),
    not viol_night(S,H).

% 3) Capacity clamps (only if not night/seasonal for solar)
repair(bound_cap, wind,  S,H) :- viol_cap(wind,S,H).

repair(bound_cap, solar, S,H) :-
    viol_cap(solar,S,H),
    not viol_night(S,H),
    not viol_seasonal(S,H).

% 4) Lower bound clamp (only if no higher-priority repair applies)

% load: only low bound matters
repair(bound_low, load, S,H) :-
    viol_low(load,S,H).

% wind: don't emit bound_low if it will be cap-repaired
repair(bound_low, wind, S,H) :-
    viol_low(wind,S,H),
    not viol_cap(wind,S,H).

% solar: don't emit bound_low if night/seasonal/cap repair applies
repair(bound_low, solar, S,H) :-
    viol_low(solar,S,H),
    not viol_night(S,H),
    not viol_seasonal(S,H),
    not viol_cap(solar,S,H).

