% src/asp/finance_physics.lp

% Targets: Close Price, Volatility (ATR)
target(close; vol).

% --- DECISION SPACE ---
1 { keep(T,S,H); repair(bound_low,T,S,H); repair(bound_sanity,T,S,H) } 1 
    :- target(T), sample(S), horizon(H), pred(T,S,H,_).

% --- RULE 1: NON-NEGATIVITY (Bankruptcy) ---
% Prices and Volatility cannot be negative.
:- keep(T,S,H), pred(T,S,H,V), V < 0.

% Trigger repair
:- pred(T,S,H,V), V < 0, not repair(bound_low,T,S,H).

% --- RULE 2: MINIMUM VOLATILITY ---
% Volatility cannot be 0.
% New Scale: 1 (Dollar). 
% We check V < 1 (Less than $1 volatility per minute is effectively zero).
:- keep(vol,S,H), pred(vol,S,H,V), V < 1.

% --- RULE 3: CIRCUIT BREAKERS (Sanity Bounds) ---
% New Scale: 1 (Dollar)

sanity_max(1000000).   % $1,000,000 Max
sanity_min(100).       % $100 Min

% Max Check
:- keep(close,S,H), 
   pred(close,S,H,V), 
   sanity_max(Max), V > Max.

% Min Check
:- keep(close,S,H), 
   pred(close,S,H,V), 
   sanity_min(Min), V < Min.

% --- OPTIMIZATION ---
:~ repair(bound_low,T,S,H).    [10@1,T,S,H]
:~ repair(bound_sanity,T,S,H). [5@1,T,S,H]

#show repair/4.