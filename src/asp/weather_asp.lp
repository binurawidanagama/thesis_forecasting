% src/asp/weather_asp.lp

% --- 1. WIND CUT-IN RULE ---
% Turbines typically cut in at 3 m/s. 
% Scaled units: 3 m/s * 100 = 300.
wind_cut_in(300).

% Hard Constraint: If wind speed is below cut-in, Power MUST be 0.
% We trigger a repair(bound_low,...) which forces it to 0.
:- keep(wind,S,H), 
   pred(wind,S,H,V), V > 0, 
   weather(wind_speed,S,H,WS), 
   wind_cut_in(Limit), WS < Limit.

% --- 2. SOLAR IRRADIANCE CONSISTENCY ---
% If GHI (Global Horizontal Irradiance) is effectively zero (thick cloud/dusk), 
% Solar power must be very low.
% GHI is in W/m^2. Let's say < 10 W/m^2 implies ~0 Output.
% Scaled: 10 * 100 = 1000.
ghi_low_threshold(1000).

:- keep(solar,S,H),
   pred(solar,S,H,V), V > 0,
   weather(ghi,S,H,G),
   ghi_low_threshold(T), G < T.

% (Optional) Betz Limit Logic / Power Curve Consistency could go here
% but Cut-In is the most impactful one for removing noise.