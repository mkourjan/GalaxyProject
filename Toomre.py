
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from IPython.html.widgets import interact, interactive
from scipy.optimize import fsolve

pi = np.pi
def ring(particles,radius,G,M):
    
    '''
    Set initial positions and velocities for all stars.
    '''
    
    particle = []
    velocity = []
    theta_n = 0
    arclen = (2*pi)/particles           ## Arc length equally divided amongst the number of particles around circle
    v = np.sqrt((G*M)/radius)           ## Velocity for central force to stay in a circular orbit
    while len(particle) < particles:
        angle = theta_n*arclen
        beta = angle + (pi/2.)          ## Angle beta = angle of the position plus 90 degrees, for velocity vector
        theta_n += 1
        particle.append((radius*np.cos(angle), radius*np.sin(angle)))   ## x = r*cos(theta)  y = r*sin(theta)
        velocity.append((v*np.cos(beta), v*np.sin(beta)))               ## vx = v*cos(beta)  vy = v*sin(beta)
    return np.array(particle),np.array(velocity)            ## Returns two arrays, particle position and velocity.

def init_rings_123(G,M):
    '''
    Arrange stars into rings located at a distance that is a 
    certain percentage of Rmin.  Rmin is the minimum distance 
    between the disruptor galaxy and the disrupted galaxy.
    This function is only used on direct, retrograde, and light
    mass disruptor cases.
    '''
    
    ring1,velocity1 = ring(12,.2,G,M)     ## The positions of the stars are dependent on details found in the paper by Toomre et al.
    ring2,velocity2 = ring(18,.3,G,M)
    ring3,velocity3 = ring(24,.4,G,M)
    ring4,velocity4 = ring(30,.5,G,M)
    ring5,velocity5 = ring(36,.6,G,M)
    rings = np.array([ring1,ring2,ring3,ring4,ring5])
    velocity = np.array([velocity1,velocity2,velocity3,velocity4,velocity5])
    return rings,velocity             ## Returns arrays of both the positions and velocity.

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
    
    '''
    Make 4 arrays that hold the information for the x and y star positions, and the
    x and y star velocities. 
    '''
    
    rx_points = []                             ## x-coordinates of all stars
    ry_points = []                             ## y-coordinates of all stars
    vrx = []                                   ## initial x velocity of all stars
    vry = []                                   ## initial y velocity of all stars
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
    
    '''
    Function extracts information from the 4 arrays in unpack_rings_vel
    function and plugs the values into the given differential equations
    that describe the motion of the stars and the motion of the disruptor
    galaxy.  The output is a single array.
    '''
    
    G = 4.302e-3 #pc(M_solar)^-1 (km/s)^2
    vRx = y[0]
    vRy = y[2]
    Rx = y[1]
    Ry = y[3]
    vrx = y[4]
    vry = y[6]
    rx = y[5]
    ry = y[7]
    R = np.sqrt(Rx**2+Ry**2)
    delta_x = (Rx-rx)
    delta_y = (Ry-ry)
    
    dvrx_dt = -G * ((M/np.sqrt(rx**2. + ry**2.)**3.)*rx - (S/np.sqrt(delta_x**2.+delta_y**2.)**3.)*delta_x #Given differential equation describing the motion of the stars.
                                                        + (S/np.sqrt(Rx**2.+Ry**2.)**3.)*Rx)
    dvry_dt = -G * ((M/np.sqrt(rx**2. + ry**2.)**3.)*ry - (S/np.sqrt(delta_x**2.+delta_y**2.)**3.)*delta_y 
                                                        + (S/np.sqrt(Rx**2.+Ry**2.)**3.)*Ry)
    
    dvRx_dt = -G * ((M+S)/(np.sqrt(Rx**2+Ry**2))**3)*Rx #Given differential equation describing the motion of the Disruptor Galaxy.
    dvRy_dt = -G * ((M+S)/(np.sqrt(Rx**2+Ry**2))**3)*Ry
    
    return np.array([dvRx_dt, vRx, dvRy_dt, vRy, dvrx_dt, vrx, dvry_dt, vry])

def Make_Master_Array(Case = 1,Rx0 = -8, Ry0 = -9,Initial_velocity_X = 0.85,Initial_velocity_Y = 0.65,t = 11., M=330., S=330., dt = 0.01):
    
    '''
    The function takes the single array from derivgalaxy and plugs it into
    the odeint solver.  The function then filters out the information associated
    with the positions of the disruptor galaxy and stars at all timesteps between
    0 and 20 with 0.0075 intervals.  The output is a 2 dimension matrix where the 
    columns represent positions of the disruptor galaxy and all of the stars, and the
    rows represent a particular time.
    '''
    
    G = 4.302e-3 #pc(M_solar)^-1 (km/s)^2\
    
    
    if Case ==1 or Case == 2 or Case == 3:
        rings,velocity = init_rings_123(G,M)    ## Chooses which function to run according to the example chosen
    elif Case == 4:
        rings,velocity = init_rings_4(G,M)
    
    
    rx0,ry0,vrx_0,vry_0 = unpack_rings_vel(rings,velocity)    ## Converts values determined above to 1-D arrays
    vRx_0 = Initial_velocity_X          ## Initial velocity of disruptor galaxy in x direction
    vRy_0 = Initial_velocity_Y          ## Initial velocity of disruptor galaxy in y direction
    
    ts = np.arange(0.,t+0.1,0.0075)
    
    MasterArray = []
    
    for n in range(len(rx0)):   ## Runs for all 120 particles in initial condition vectors.
        
        output = odeint(derivgalaxy, np.array([vRx_0,Rx0,vRy_0,Ry0,vrx_0[n],rx0[n],vry_0[n],ry0[n]]),
                        ts, args=(M, S)) ## Solve the differential equation for each time index 
                                         ##and output the position values of the stars and disruptor galaxy.
            
        
        rx = output[:,5]                
        ry = output[:,7]
            
        if n == 0:
            
            Rx = output[:,1] 
            Ry = output[:,3]                
            
            MasterArray.append(Rx)
            MasterArray.append(Ry)
            MasterArray.append(rx)
            MasterArray.append(ry)
                        
            
        else:
            MasterArray.append(rx)
            MasterArray.append(ry)
            
    return MasterArray

def Make_Plots(results,t, dt):
    
    '''
    Function extracts all positions of stars and the disruptor galaxy from a matrix and plots them.
    This is the Direct Passage situation.
    '''
    
    index = int(t/dt)
    plt.figure(figsize = (7, 7))
    plt.grid()
    plt.xlim(-10,7)
    plt.ylim(-16,15)
    plt.plot(results[0][:index], results[1][:index], 'b--', label = 'Disturbant Galaxy')
    for i in range(1,121):
        plt.plot(results[2*i][index], results[2*i + 1][index], 'ro', label = "Stars")
    plt.show()
    
def Make_Plots_Green_Star(results, t, dt, GreenStar):
    index = int(t/dt)
    plt.figure(figsize = (7, 7))
    plt.grid()
    plt.xlim(-10,7)
    plt.ylim(-16,15)
    plt.plot(results[0][:index], results[1][:index], 'b--', label = 'Disturbant Galaxy')
    for i in range(1,121):
        plt.plot(results[2*i][index], results[2*i + 1][index], 'ro', label = "Stars")
    for i in range(GreenStar, GreenStar+1):
        plt.plot(results[2*i][index], results[2*i + 1][index], 'go', label = "Highlighted Star")
    plt.show()
    
    
def Generate_Data(dataset = 'all', save = True):
    
    '''
    Calculate data for all of the stars and disruptor galaxy at all timesteps.
    '''
    
    if dataset == 'all':

        results_A = Make_Master_Array(Case = 1, Rx0 = -8, Ry0 = -9, Initial_velocity_X = 0.85,Initial_velocity_Y = 0.65,t = 20, M=330., S=330., dt = 0.0075)
        #Direct Passage 

        results_B = Make_Master_Array(Case = 2, Rx0 = -8, Ry0 = 9,Initial_velocity_X = 0.85,Initial_velocity_Y = -0.65,t = 20, M=330., S=330., dt = 0.0075)
        #Retrograde Passage

        results_C = Make_Master_Array(Case = 3,Rx0 = -8, Ry0 = -9, Initial_velocity_X = 0.85,Initial_velocity_Y = 0.65,t = 20, M=330., S=82.5, dt = 0.0075)
        #Light Mass Disruptor

        results_D = Make_Master_Array(Case = 4, Rx0 = -8, Ry0 = -9, Initial_velocity_X = 0.85,Initial_velocity_Y = 0.65,t = 20, M=82.5, S=330., dt = 0.0075)
        #Heavy Mass Disruptor

    if save == True:
        
        np.save('Toomre_A.npy', results_A)
        np.save('Toomre_B.npy', results_B)
        np.save('Toomre_C.npy', results_C)
        np.save('Toomre_D.npy', results_D)