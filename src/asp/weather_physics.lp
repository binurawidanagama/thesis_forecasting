% src/asp/weather_physics.lp

target(temp; hum; wind; press; rad; precip).

% --- DECISION SPACE ---
% We added: repair(noise_gate,...) and repair(boost_hum,...)
1 { 
    keep(T,S,H); 
    repair(bound,T,S,H); 
    repair(night,T,S,H); 
    repair(noise_gate,T,S,H); 
    repair(boost_hum,T,S,H)
} 1 :- target(T), sample(S), horizon(H), pred(T,S,H,_).

% --- 1. HUMIDITY ---
% Bounds: 0% to 100% (Scale: 100 -> 0 to 10000)
:- keep(hum,S,H), pred(hum,S,H,V), V < 0.
:- keep(hum,S,H), pred(hum,S,H,V), V > 10000. 

% NEW: Consistency - If raining heavily, Humidity cannot be dry.
% Precip > 0.5mm (50) implies Humidity >= 60% (6000)
:- keep(hum,S,H), 
   pred(precip,S,H,P), P > 50, 
   pred(hum,S,H,H_val), H_val < 6000.

% --- 2. PRECIPITATION ---
% Bounds
:- keep(precip,S,H), pred(precip,S,H,V), V < 0.

% NEW: Noise Gate - Tiny precip is likely noise.
% 0 < P < 0.1mm (10)
:- keep(precip,S,H), pred(precip,S,H,V), V > 0, V < 10.

% --- 3. TEMPERATURE ---
% NEW: Regional Bounds (Austria: -30C to +45C)
% Scale 100: -3000 to 4500
:- keep(temp,S,H), pred(temp,S,H,V), V < -3000.
:- keep(temp,S,H), pred(temp,S,H,V), V > 4500.

% --- 4. RADIATION (Existing Night Rule) ---
:- keep(rad,S,H), pred(rad,S,H,V), V < 0.
:- keep(rad,S,H), night(S,H), pred(rad,S,H,V), V > 0.

% --- 5. OTHERS (Standard) ---
:- keep(wind,S,H), pred(wind,S,H,V), V < 0.
:- keep(press,S,H), pred(press,S,H,V), V < 85000.
:- keep(press,S,H), pred(press,S,H,V), V > 110000.

% --- LOGIC GUARDS ---
% Prevent "bound" repair on valid positive numbers
:- repair(bound,T,S,H), pred(T,S,H,V), V >= 0, T != temp. 
% Prevent "night" repair during day
:- repair(night,T,S,H), not night(S,H).

% --- COSTS (Optimization) ---
% We prefer small fixes (noise gate) over big logic changes (boost_hum)
:~ repair(bound,T,S,H).      [10@1,T,S,H]
:~ repair(night,T,S,H).      [10@1,T,S,H]
:~ repair(noise_gate,T,S,H). [1@1,T,S,H]  % Cheap to fix noise
:~ repair(boost_hum,T,S,H).  [5@1,T,S,H]  % More expensive to alter physics

#show repair/4.