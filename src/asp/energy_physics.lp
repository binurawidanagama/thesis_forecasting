% =====================================================================
% src/asp/energy_physics.lp  (FAST deterministic repairs)
%
% Facts expected (integers scaled by SCALE=100 in Python):
%   sample(S).                      % S is local per batch: 0..B-1
%   horizon(H).                     % 1..Hz
%   pred(wind|solar|load, S, H, V). % scaled int
%   night(S,H).                     % optional
%   month(S,M).                     % optional (1..12) for seasonal solar ceiling
%   cap(wind,S,C). cap(solar,S,C).  % optional capacity caps (scaled int)
%
% Output:
%   repair(kind, target, sample, horizon).
%
% Kinds:
%   bound      -> clamp to >=0 (all targets) OR pressure-style upper/lower if you add later
%   cap        -> clamp to capacity (wind/solar) if cap facts provided
%   night      -> force solar=0 at night
%   seasonal   -> clamp solar to monthly CF ceiling if month+cap provided
% =====================================================================

target(wind; solar; load).

% ---------------------------
% Violations
% ---------------------------

% Non-negativity for all
viol_low(T,S,H) :- target(T), pred(T,S,H,V), V < 0.

% Capacity ceilings (only if cap facts exist)
viol_cap(wind,S,H)  :- pred(wind,S,H,V),  cap(wind,S,C),  V > C.
viol_cap(solar,S,H) :- pred(solar,S,H,V), cap(solar,S,C), V > C.

% Night rule: solar must be 0 at night (fix also negatives/positives)
viol_night(S,H) :- night(S,H), pred(solar,S,H,V), V != 0.

% Seasonal solar ceiling via month & capacity (optional but nice)
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

viol_seasonal(S,H) :-
    pred(solar,S,H,V),
    cap(solar,S,C),
    month(S,M),
    max_solar_cf(M,LimitPct),
    Allowed = (C * LimitPct) / 100,
    V > Allowed.

% ---------------------------
% Repairs with precedence (one repair per cell)
% Precedence:
%   solar night -> night
%   solar seasonal -> seasonal
%   cap -> cap
%   negative -> bound
% ---------------------------

% 1) Night solar dominates
repair(night, solar, S,H) :- viol_night(S,H).

% 2) Seasonal clamp for solar (only if not night repair)
repair(seasonal, solar, S,H) :-
    viol_seasonal(S,H),
    not viol_night(S,H).

% 3) Capacity clamps (only if not night/seasonal for solar)
repair(cap, wind,  S,H) :- viol_cap(wind,S,H).
repair(cap, solar, S,H) :-
    viol_cap(solar,S,H),
    not viol_night(S,H),
    not viol_seasonal(S,H).

% 4) Lower bound clamp
repair(bound, T, S,H) :-
    viol_low(T,S,H),
    % if solar is night/seasonal/cap repaired, don't also emit bound
    not (T=solar, viol_night(S,H)),
    not (T=solar, viol_seasonal(S,H)),
    not (T=solar, viol_cap(solar,S,H)),
    not (T=wind,  viol_cap(wind,S,H)).

#show repair/4.
