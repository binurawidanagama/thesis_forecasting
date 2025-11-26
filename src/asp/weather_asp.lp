% src/asp/weather_asp.lp

% ---------------------------------------------------------
% RULE 1: WIND CUT-IN SPEED
% ---------------------------------------------------------
% Physics: A wind turbine generally needs 3 m/s to overcome friction.
% Scaled: 3.0 m/s * 100 = 300.
wind_cut_in_threshold(300).

% Constraint: It is illegal to have Wind Power > 0.1 MW if Wind Speed < 3 m/s.
:- keep(wind,S,H),
   pred(wind,S,H,V), V > 10,  % Allow tiny noise (0.1 MW)
   weather(wind_speed,S,H,WS),
   wind_cut_in_threshold(Limit),
   WS < Limit.

% ---------------------------------------------------------
% RULE 2: WIND POWER CURVE ENVELOPE
% ---------------------------------------------------------
% Physics: If wind < 6 m/s, Output cannot exceed 20% of Capacity.
% (6 m/s * 100 = 600).
wind_low_speed(600).
wind_low_power_ratio(20). % 20%

:- keep(wind,S,H),
   weather(wind_speed,S,H,WS),
   wind_low_speed(W_Limit), WS < W_Limit,
   pred(wind,S,H,V),
   cap(wind,S,C),
   wind_low_power_ratio(Ratio),
   Allowed = (C * Ratio) / 100,
   V > Allowed.

% --- SOLAR RULES REMOVED TO RESTORE SMOOTH CURVE ---