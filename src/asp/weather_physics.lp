% =====================================================================
% src/asp/weather_physics.lp  (FAST deterministic repairs)
%
% Expected facts (integers scaled by SCALE=100 in Python):
%   pred(temp|hum|wind|press|rad|precip, S, H, V).
%   night(S,H).         % provided by python
%
% Emits:
%   repair(kind, target, sample, horizon).
% kinds: bound, night, noise_gate, boost_hum
% =====================================================================

target(temp; hum; wind; press; rad; precip).

% ---------------------------
% Violations (scaled ints)
% ---------------------------

% humidity 0..100% => 0..10000
viol_hum_low(S,H)  :- pred(hum,S,H,V), V < 0.
viol_hum_high(S,H) :- pred(hum,S,H,V), V > 10000.

% precip must be >=0
viol_precip_low(S,H) :- pred(precip,S,H,V), V < 0.

% drizzle noise gate: 0 < P < 0.1mm => 0 < V < 10
viol_precip_noise(S,H) :- pred(precip,S,H,V), V > 0, V < 10.

% temp Austria bounds [-30,45] => [-3000,4500]
viol_temp_low(S,H)  :- pred(temp,S,H,V), V < -3000.
viol_temp_high(S,H) :- pred(temp,S,H,V), V > 4500.

% radiation must be >=0, and must be 0 at night
viol_rad_low(S,H)   :- pred(rad,S,H,V), V < 0.
viol_rad_night(S,H) :- night(S,H), pred(rad,S,H,V), V != 0.

% wind must be >=0
viol_wind_low(S,H) :- pred(wind,S,H,V), V < 0.

% pressure 850..1100 hPa => 85000..110000
viol_press_low(S,H)  :- pred(press,S,H,V), V < 85000.
viol_press_high(S,H) :- pred(press,S,H,V), V > 110000.

% raining consistency: if precip >0.5mm => P>50 then hum>=60% => >=6000
viol_boost_hum(S,H) :-
    pred(precip,S,H,P), P > 50,
    pred(hum,S,H,Hv), Hv < 6000.

% ---------------------------
% Repairs with precedence (one repair per cell)
% Precedence idea:
%   rad at night -> night repair
%   precip noise -> noise_gate
%   humidity consistency -> boost_hum (only if not already out-of-range)
%   everything else -> bound
% ---------------------------

% 1) Radiation night rule (dominant for rad)
repair(night, rad, S,H) :- viol_rad_night(S,H).

% 2) Precip noise gate
repair(noise_gate, precip, S,H) :- viol_precip_noise(S,H).

% 3) Boost humidity if raining (but don't do this if humidity is already invalid range)
repair(boost_hum, hum, S,H) :-
    viol_boost_hum(S,H),
    not viol_hum_low(S,H),
    not viol_hum_high(S,H).

% 4) Bounds repair (everything else)
repair(bound, hum,   S,H) :- viol_hum_low(S,H).
repair(bound, hum,   S,H) :- viol_hum_high(S,H).
repair(bound, precip,S,H) :- viol_precip_low(S,H).
repair(bound, temp,  S,H) :- viol_temp_low(S,H).
repair(bound, temp,  S,H) :- viol_temp_high(S,H).
repair(bound, rad,   S,H) :- viol_rad_low(S,H), not viol_rad_night(S,H).
repair(bound, wind,  S,H) :- viol_wind_low(S,H).
repair(bound, press, S,H) :- viol_press_low(S,H).
repair(bound, press, S,H) :- viol_press_high(S,H).

#show repair/4.
