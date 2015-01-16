
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from IPython.html.widgets import interact, interactive, fixed

G = 4.49955370898e-08 #kpc^3 Msol^-1 (10^8 years)^-2
RCM = 0.; xCM = 0.; yCM = 0. #let origin be at CM
Rmin = 25 #kpc
pi = np.pi

def Set_Init_R_Cond(Ry0, M, S):
    #minimum separation distance
    Rmin = 25.0 #kpc

    #Velocity at distance of closest approach
    Vmin = np.sqrt(2.*G*(M+S)/Rmin) #parabolic orbit

    #Angular momentum per unit mass at distance of closest approach
    hmin = Rmin*Vmin #r x v - angular momentum per unit mass 

    #From the orbital equations, relationship between geometric and physical parameters
    c = hmin**2/(G*(M+S))
    beta = c/2.
    alpha = -1./(2*c)

    #Set a range of y values and compute the corresponding x,R values for initial points on a parabolic orbit
    
    Rx0 = beta + alpha*Ry0**2

    R0 = np.sqrt(Rx0**2+Ry0**2)

    #unit tangent vector for the parabola
    T_x = 2*alpha*Ry0/np.sqrt((2*alpha*Ry0)**2+1)
    T_y = 1./np.sqrt((2*alpha*Ry0)**2+1)

    #Velocity vector
    V0 = np.sqrt(2.*G*(M+S)/R0)
    Vx0 = V0*T_x
    Vy0 = V0*T_y
    
    if (Ry0>0.):
        Vx0 *= -1.
        Vy0 *= -1.
    return Rx0, Ry0, Vx0, Vy0

def ring(particles,percent,G,M):
    
    radius = Rmin * percent
    particle = []
    velocity = []
    theta_n = 0
    arclen = (2*pi)/particles              ## Arc length equally divided amongst the number of particles around circle
    v = np.sqrt((G*M)/radius)                 ## Velocity for central force to stay in a circular orbit
    while len(particle) < particles:
        angle = theta_n*arclen
        beta = angle + (pi/2.)          ## Angle beta = angle of the position plus 90 degrees, for velocity vector
        theta_n += 1
        particle.append((radius*np.cos(angle), radius*np.sin(angle)))   ## x = r*cos(theta)  y = r*sin(theta)
        velocity.append((v*np.cos(beta), v*np.sin(beta)))               ## Same concept here as above.
    return np.array(particle),np.array(velocity)            ## Returns two arrays, particle position and velocity.

def init_rings_123(G,M):
    ring1,velocity1 = ring(12,.2,G,M)     ## All of these are dependent on details found in the paper by Toomre et al.
    ring2,velocity2 = ring(18,.3,G,M)
    ring3,velocity3 = ring(24,.4,G,M)
    ring4,velocity4 = ring(30,.5,G,M)
    ring5,velocity5 = ring(36,.6,G,M)
    rings = np.array([ring1,ring2,ring3,ring4,ring5])
    velocity = np.array([velocity1,velocity2,velocity3,velocity4,velocity5])
    return rings,velocity             ## Returns arrays of both

def init_rings_4(G,M):
    
    '''
    Arrange stars into rings located at a distance that is a 
    certain percentage of Rmin. Rmin is the minimum distance 
    between the disruptor galaxy and the disrupted galaxy. This 
    function is only used on the heavy mass disruptor case.
    '''
    
    ring1,velocity1 = ring(12,.12,G,M)     ## The positions of the stars are dependent on details found in the paper by Toomre et al.
    ring2,velocity2 = ring(18,.18,G,M)
    ring3,velocity3 = ring(24,.24,G,M)
    ring4,velocity4 = ring(30,.30,G,M)
    ring5,velocity5 = ring(36,.36,G,M)
    rings = np.array([ring1,ring2,ring3,ring4,ring5])
    velocity = np.array([velocity1,velocity2,velocity3,velocity4,velocity5])
    return rings,velocity             ## Returns arrays of both the positions and velocity.

def unpack_rings_vel(rings,velocity):
    rx_points = []                             ## x-coordinates of all massless particles
    ry_points = []                             ## y-coordinates
    vrx = []                                   ## initial x velocity
    vry = []                                   ## initial y velocity
    for ring in rings:
        for point in ring:
            rx_points.append(point[0])
            ry_points.append(point[1])
    for ring in velocity:
        for point in ring:
            vrx.append(point[0])
            vry.append(point[1])
    return np.array(rx_points), np.array(ry_points), np.array(vrx), np.array(vry)  ## Returns numpy arrays of values

def derivgalaxy(y,t,M,S):
    G = 4.49955370898e-08 #kpc^3 M_sol^-1 unitTime^-2
    Rx = y[0]
    Vx = y[1]

    Ry = y[2]
    Vy = y[3]
    
    rxs = y[4]
    vrx = y[5]
    
    rys = y[6]
    vry = y[7]
    
    delta_x = (Rx-rxs)
    delta_y = (Ry-rys)
    
    R = np.sqrt(Rx**2+Ry**2)
    
    dvrx_dt = -G * ((M/np.sqrt(rxs**2. + rys**2.)**3.)*rxs - (S/np.sqrt(delta_x**2.+delta_y**2.)**3.)*delta_x 
                                                        + (S/np.sqrt(Rx**2.+Ry**2.)**3.)*Rx)
    dvry_dt = -G * ((M/np.sqrt(rxs**2. + rys**2.)**3.)*rys - (S/np.sqrt(delta_x**2.+delta_y**2.)**3.)*delta_y 
                                                        + (S/np.sqrt(Rx**2.+Ry**2.)**3.)*Ry)
    
    dvRx_dt = -G * ((M+S)/(np.sqrt(Rx**2+Ry**2))**3)*Rx                                                      
    dvRy_dt = -G * ((M+S)/(np.sqrt(Rx**2+Ry**2))**3)*Ry
    
    return np.array([Vx, dvRx_dt, Vy, dvRy_dt, vrx, dvrx_dt, vry, dvry_dt])

def Make_Master_Array(Case = 1, Rx0 = -39., Ry0 = -80.,M=1.0e11, S=1.0e11):
    
    '''
    The function takes the single array from derivgalaxy and plugs it into
    the odeint solver.  The function then filters out the information associated
    with the positions of the disruptor galaxy and stars at all timesteps between
    0 and 20 with 0.0075 intervals.  The output is a 2 dimension matrix where the 
    columns represent positions of the disruptor galaxy and all of the stars, and the
    rows represent a particular time.
    '''
    Rmin = 25 #kpc
    tmax = 20.
    dt = 0.007
    ts = np.arange(0.,tmax+dt/10.,dt)
    
    
    if Case ==1 or Case == 2 or Case == 3:
        rings,velocity = init_rings_123(G,M)    ## Chooses which function to run according to the example chosen
    elif Case == 4:
        rings,velocity = init_rings_4(G,M)
    
    Rx0, Ry0, Vx0, Vy0 = Set_Init_R_Cond(Ry0=Ry0, M = M, S = S)
    rx0,ry0,vx0,vy0 = unpack_rings_vel(rings,velocity)    ## Converts values determined above to 1-D arrays
    
    
    MasterArray = []
    
    for n in range(len(rx0)):   ## Runs for all 120 particles in initial condition vectors.
        
        output = odeint(derivgalaxy, np.array([Rx0, Vx0, Ry0, Vy0, rx0[n], vx0[n],ry0[n], vy0[n]]),
                        ts, args=(M, S)) ## Solve the differential equation for each time index 
                                         ##and output the position values of the stars and disruptor galaxy.
            
        
        rx = output[:,4]                
        ry = output[:,6]
            
        if n == 0:
            
            Rx = output[:,0] 
            Ry = output[:,2]                
            
            MasterArray.append(Rx)
            MasterArray.append(Ry)
            MasterArray.append(rx)
            MasterArray.append(ry)
                        
            
        else:
            MasterArray.append(rx)
            MasterArray.append(ry)
            
    return MasterArray

def Make_Plot_stars(results, M, S, t, dt):

    index = int(t/dt)
    
    Rx = results[0][:index]
    Ry = results[1][:index]
    RxS = xCM + (M/(M+S))*Rx
    RyS = yCM + (M/(M+S))*Ry
    RxM = xCM - (S/(M+S))*Rx
    RyM = yCM - (S/(M+S))*Ry
    RxS -= RxM
    RyS -= RyM
    RxM -= RxM
    RyM -= RyM
    plt.plot(RxS, RyS, 'b--', label = 'Disturbing Galaxy')
    plt.plot(RxS[-1], RyS[-1], 'bo')
    plt.plot(RxM, RyM, 'r--', label = 'Main Galaxy')
    plt.plot(RxM[-1], RyM[-1], 'ro')
    for i in range(1, 121):
        plt.plot(results[2*i][index]+RxM[-1], results[2*i + 1][index]+RyM[-1], 'go', label = "Stars")
        
    plt.xlim(1.1*Rx[0],xCM-1.1*Rx[0])
    plt.ylim(1.1*Ry[0],yCM-1.1*Ry[0])
    plt.xlim(-150,150)
    plt.ylim(-150,150)
    plt.grid()
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          #ncol=2, fancybox=True, shadow=True)
    plt.show()
    
def Make_Plots_Yellow_Star(results, M, S, t, dt, YellowStar):
    
    index = int(t/dt)
    
    Rx = results[0][:index]
    Ry = results[1][:index]
    RxS = xCM + (M/(M+S))*Rx
    RyS = yCM + (M/(M+S))*Ry
    RxM = xCM - (S/(M+S))*Rx
    RyM = yCM - (S/(M+S))*Ry
    RxS -= RxM
    RyS -= RyM
    RxM -= RxM
    RyM -= RyM
    plt.plot(RxS, RyS, 'b--', label = 'Disturbing Galaxy')
    plt.plot(RxS[-1], RyS[-1], 'bo')
    plt.plot(RxM, RyM, 'r--', label = 'Main Galaxy')
    plt.plot(RxM[-1], RyM[-1], 'ro')
    for i in range(1, 121):
        plt.plot(results[2*i][index]+RxM[-1], results[2*i + 1][index]+RyM[-1], 'go', label = "Stars")
    for i in range(YellowStar, YellowStar+1):
        plt.plot(results[2*i][index], results[2*i + 1][index], 'yo', label = "Highlighted Star")    
    #plt.xlim(1.1*x[0],xCM-1.1*x[0])
    #plt.ylim(1.1*y[0],yCM-1.1*y[0])
    plt.xlim(-150,150)
    plt.ylim(-150,150)
    plt.grid()
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          #ncol=2, fancybox=True, shadow=True)
    plt.show()
    
def Generate_Data(dataset = 'all', save = True):
    
    '''
    Calculate data for all of the stars and disruptor galaxy at all timesteps.
    '''
    
    if dataset == 'all':

        results_A = Make_Master_Array(Case = 1, Rx0 = -39., Ry0 = -80., M=1.0e11, S=1.0e11)       
        #Direct Passage 

        results_B = Make_Master_Array(Case = 2, Rx0 = -39., Ry0 = 80., M=1.0e11, S=1.0e11)                                                                                
        #Retrograde Passage

        results_C = Make_Master_Array(Case = 3, Rx0 = -39., Ry0 = -80., M=1.0e11, S=1.0e11/4)
        #Light Mass Disruptor

        results_D = Make_Master_Array(Case = 4, Rx0 = -39., Ry0 = -80., M=1.0e11, S=1.0e11*4)
        #Heavy Mass Disruptor

    if save == True:
        
        np.save('Toomre_A.npy', results_A)
        np.save('Toomre_B.npy', results_B)
        np.save('Toomre_C.npy', results_C)
        np.save('Toomre_D.npy', results_D)    
    