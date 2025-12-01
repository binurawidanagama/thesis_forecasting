% src/asp/weather_physics.lp

% Targets: temp, humidity, wind, pressure, radiation, precipitation
target(temp; hum; wind; press; rad; precip).

% --- DECISION SPACE ---
1 { keep(T,S,H); repair(bound,T,S,H); repair(night,T,S,H) } 1 
    :- target(T), sample(S), horizon(H), pred(T,S,H,_).

% --- PHYSICS RULES ---

% 1. HUMIDITY: 0% to 100%
:- keep(hum,S,H), pred(hum,S,H,V), V < 0.
:- keep(hum,S,H), pred(hum,S,H,V), V > 10000. 

% 2. WIND: Non-negative
:- keep(wind,S,H), pred(wind,S,H,V), V < 0.

% 3. PRESSURE: 850hPa to 1100hPa
:- keep(press,S,H), pred(press,S,H,V), V < 85000.
:- keep(press,S,H), pred(press,S,H,V), V > 110000.

% 4. PRECIPITATION: Non-negative
:- keep(precip,S,H), pred(precip,S,H,V), V < 0.

% 5. RADIATION: Non-negative AND Night Rule
:- keep(rad,S,H), pred(rad,S,H,V), V < 0.

% Night Rule: If night, Radiation must be 0
:- keep(rad,S,H), 
   night(S,H), 
   pred(rad,S,H,V), 
   V > 0.

% --- COST ---
:~ repair(bound,T,S,H). [1@1,T,S,H]
:~ repair(night,T,S,H). [1@1,T,S,H]

#show repair/4.