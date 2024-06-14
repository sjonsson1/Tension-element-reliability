import numpy as np

def F_morison_swell_current(significant_wave_height_swell,
                            U_current_veloctiy):
    # General properties (assumes cylindrical geometry; SI units)
    rhoW = 1025
    D = np.sqrt(38/np.pi)*2
    depth = 30

    # Swell Wave properties
    Tp = 12
    Omega = 2*np.pi/Tp
    L = 9.81*Tp**2/(2*np.pi)
    k = 2*np.pi/L
    Amplitude = significant_wave_height_swell/2

    
    # Water velocity
    cos_func = 1 #np.cos(omega*t-k*x)
    u_swell =  Amplitude*Omega*np.exp(-k*depth)*cos_func
    
    # Intertia force (with mass coefficient)
    CM = 1+((9.2/2) / (20.8/2))
    F_inertia_swell =  (np.pi*rhoW*CM*D**2)/4 * (u_swell)
    
    # Drag force (with drag coefficient)
    CD = 0.13
    F_drag_swell = 0.5*rhoW*CD*D*(u_swell+U_current_veloctiy)**2
    
    F_combined_swell_current = np.sqrt(F_inertia_swell**2 + F_drag_swell**2)
    
    return F_combined_swell_current


def F_morison_all(significant_wave_height_swell, U_current_veloctiy, significant_wave_height_windsea):
    # General properties (assumes cylindrical geometry; SI units)
    rhoW = 1025
    D = np.sqrt(38/np.pi)*2
    depth = 30

    # Wind sea Wave properties
    Tp = 12
    Omega = 2*np.pi/Tp
    L = 9.81*Tp**2/(2*np.pi)
    k = 2*np.pi/L
    Amplitude = significant_wave_height_windsea/2
    
    # Water velocity
    cos_func = 1 #np.cos(omega*t-k*x)
    u_wind =  Amplitude*Omega*np.exp(-k*depth)*cos_func
    
    # Intertia force (with mass coefficient)
    CM = 1+((9.2/2) / (20.8/2))
    F_inertia_wind =  (np.pi*rhoW*CM*D**2)/4 * (u_wind)
    
    # Drag force
    CD = 0.13 # drag coefficient
    F_drag_wind = 0.5*rhoW*CD*D*(u_wind)**2
    
    F_combined_wind = np.sqrt(F_inertia_wind**2 + F_drag_wind**2)
        
    rhoW = 1025
    D = np.sqrt(38/np.pi)*2
    depth = 30

    # Swell Wave properties
    Tp = 12
    Omega = 2*np.pi/Tp
    L = 9.81*Tp**2/(2*np.pi)
    k = 2*np.pi/L
    Amplitude = significant_wave_height_swell/2

    
    # Water velocity
    cos_func = 1 #np.cos(omega*t-k*x)
    u_swell =  Amplitude*Omega*np.exp(-k*depth)*cos_func
    
    # Intertia force (with mass coefficient)
    CM = 1+((9.2/2) / (20.8/2))
    F_inertia_swell =  (np.pi*rhoW*CM*D**2)/4 * (u_swell)
    
    # Drag force (with drag coefficient)
    CD = 0.13
    F_drag_swell = 0.5*rhoW*CD*D*(u_swell+U_current_veloctiy)**2
    
    F_combined_swell_current = np.sqrt(F_inertia_swell**2 + F_drag_swell**2)

    if U_current_veloctiy > 0:
        F_tot = F_combined_swell_current + F_combined_wind
    
    else:
        F_tot = -1* (F_combined_swell_current + F_combined_wind)

    return F_tot


