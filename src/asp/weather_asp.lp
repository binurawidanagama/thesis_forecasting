% asp/weather_asp.lp


% if later you scale GHI into W/m^2 * scale, do the same here
ghi_min_for_pv(20000).   % 200.0 * 100
pv_min_check(500).       % 5.0 MW * 100

pv_weather_inconsistent(S,H) :-
    pred(solar,S,H,PV),
    pv_min_check(Pth), PV > Pth,
    ghi(S,H,G), ghi_min_for_pv(Gmin), G < Gmin.

#show pv_weather_inconsistent/2.
