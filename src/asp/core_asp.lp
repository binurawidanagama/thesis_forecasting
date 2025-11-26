% src/asp/core_asp.lp

% Define Targets
target(wind; solar; load; price).

% ---------------------------------------------------------
% 1. DECISION SPACE (The "Brain")
% ---------------------------------------------------------
% For every prediction (except price), choose exactly ONE action:
% Option A: Keep the original value (Ideal).
% Option B: Apply a specific repair if a rule is violated.
1 { keep(T,S,H); 
    repair(bound_low,T,S,H); 
    repair(bound_cap,T,S,H); 
    repair(pv_night,T,S,H); 
    repair(seasonal_cap,T,S,H) 
} 1 :- target(T), T != price, sample(S), horizon(H), pred(T,S,H,_).

% Price has a simpler logic (just bounds check)
1 { keep(price,S,H); repair(bound_low,price,S,H); repair(bound_cap,price,S,H) } 1
    :- sample(S), horizon(H), pred(price,S,H,_).

% ---------------------------------------------------------
% 2. BASIC PHYSICAL BOUNDS
% ---------------------------------------------------------
% Rule: Energy cannot be negative.
% If 'keep', value must be >= 0. If < 0, must repair.
:- keep(T,S,H), pred(T,S,H,V), V < 0.

% Rule: Energy cannot exceed Installed Capacity.
:- keep(wind,S,H),  pred(wind,S,H,V),  cap(wind,S,C),  V > C.
:- keep(solar,S,H), pred(solar,S,H,V), cap(solar,S,C), V > C.

% ---------------------------------------------------------
% 3. NIGHT RULE (Astronomy)
% ---------------------------------------------------------
% Rule: If it is night (based on hour), Solar must be 0.
% night(S,H) is passed from Python.
:- keep(solar,S,H), 
   night(S,H), 
   pred(solar,S,H,V), 
   V > 0.

% ---------------------------------------------------------
% 4. SEASONAL SOLAR CEILING (Geography)
% ---------------------------------------------------------
% Rule: In Winter, the sun angle limits maximum possible output.
% We define a "Max Capacity Factor %" per month.
% (100 = 100% of capacity allowed).

max_solar_cf(1, 40).  % Jan: Max 40%
max_solar_cf(2, 55).  % Feb: Max 55%
max_solar_cf(3, 80).  % Mar
max_solar_cf(4, 95).  % Apr
max_solar_cf(5, 100). % May
max_solar_cf(6, 100). % Jun
max_solar_cf(7, 100). % Jul
max_solar_cf(8, 95).  % Aug
max_solar_cf(9, 85).  % Sep
max_solar_cf(10, 65). % Oct
max_solar_cf(11, 45). % Nov
max_solar_cf(12, 35). % Dec

% Constraint: If we keep the prediction, it must be below the monthly limit.
:- keep(solar,S,H),
   pred(solar,S,H,V),
   cap(solar,S,C),
   month(S,M),
   max_solar_cf(M, LimitPct),
   Allowed = (C * LimitPct) / 100,
   V > Allowed.

% Trigger: Explicitly force a repair if this rule is broken
:- pred(solar,S,H,V), 
   cap(solar,S,C), 
   month(S,M), 
   max_solar_cf(M, LimitPct),
   Allowed = (C * LimitPct) / 100,
   V > Allowed,
   not repair(seasonal_cap, solar, S, H).

% ---------------------------------------------------------
% 5. OPTIMIZATION (Cost Function)
% ---------------------------------------------------------
% We want to apply the minimum number of repairs necessary.
% We assign a "cost" to each repair type.
% The solver finds the solution with the lowest total cost.

:~ repair(bound_low,T,S,H).    [10@1,T,S,H]
:~ repair(bound_cap,T,S,H).    [10@1,T,S,H]
:~ repair(pv_night,T,S,H).     [5@1,T,S,H]
:~ repair(seasonal_cap,T,S,H). [5@1,T,S,H]

#show repair/4.