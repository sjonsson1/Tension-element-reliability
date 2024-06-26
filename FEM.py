import numpy as np
import math
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import openturns as ot
from scipy import optimize



import scipy.linalg as scp
# import matplotlib.pylab as plt
import openturns.viewer as viewer
ot.Log.Show(ot.Log.NONE)

import time
import scipy

from scipy.optimize import fsolve # for plotting the LSF
from matplotlib.pylab import rcParams




# ----------------------- FEM model  ----------------------- #

def Beam2DMatrices(m, EA, EI, NodeCoord):
# Inputs:
# m         - mass per unit length [kg/m]
# EA        - axial stiffness [N]
# EI        - bending stiffness [N.m2]
# NodeCoord - ([xl, yl], [xr, yr])      
#           - left (l) and right (r) node coordinates

    # 1 - calculate length of beam (L) and orientation alpha
    xl = NodeCoord[0][0]    # x-coordinate of left node
    yl = NodeCoord[0][1]    # y-coordinate of left node
    xr = NodeCoord[1][0]    # x-coordinate of right node
    yr = NodeCoord[1][1]    # y-coordinate of rigth node
    L = np.sqrt((xr - xl)**2 + (yr - yl)**2)    # length
    alpha = math.atan2(yr - yl, xr - xl)        # angle

    # 2 - calculate transformation matrix T
    C = np.cos(alpha)
    S = np.sin(alpha)
    T = np.array([[C, S, 0], [-S, C, 0], [0, 0, 1]])
    T = np.asarray(np.bmat([[T, np.zeros((3,3))], [np.zeros((3, 3)), T]]))
    

    # 3 - calculate local stiffness and matrices
    L2 = L*L
    L3 = L*L2
    K = np.array([[EA/L, 0, 0, -EA/L, 0, 0], 
                    [0, 12*EI/L3, 6*EI/L2, 0, -12*EI/L3, 6*EI/L2], 
                    [0, 6*EI/L2, 4*EI/L, 0, -6*EI/L2, 2*EI/L], 
                    [-EA/L, 0, 0, EA/L, 0, 0], 
                    [0, -12*EI/L3, -6*EI/L2, 0, 12*EI/L3, -6*EI/L2], 
                    [0, 6*EI/L2, 2*EI/L, 0, -6*EI/L2, 4*EI/L]])
    M = m*L/420*np.array([[140, 0, 0, 70, 0, 0], 
                            [0, 156, 22*L, 0, 54, -13*L], 
                            [0, 22*L, 4*L2, 0, 13*L, -3*L2], 
                            [70, 0, 0, 140, 0, 0], 
                            [0, 54, 13*L, 0, 156, -22*L], 
                            [0, -13*L, -3*L2, 0, -22*L, 4*L2]])
    

    # 4 - rotate the matrices
    # K = K
    # M = M
    # 4 - rotate the matrices
    K = np.matmul(np.transpose(T), np.matmul(K, T))
    M = np.matmul(np.transpose(T), np.matmul(M, T))
    return M, K, alpha, T

# ----------------------- Beam Material Properties ----------------------- #
# Properties of the beam
a = 12.6                           #m
b = 12.6                            #m
A = (12.6/2)**2*3.14                             #m2
E_c = 50e9                          #E-modulus of high strength concrete
J = (np.pi*a**3*b**3)/(a**2 + b**2) #Torsion constant of the beam
G = E_c / (2*(1+0.3))
rho_c = 2500                        #kg/m^3 of concrete

print(f'Height of the beam: {a:.2e} m')
print(f'Width of the beam: {b:.2e} m')
print(f'Area of the beam: {A:.2e} m2')
print(f'E-modulus of the beam: {E_c:.2e} Pa')
print(f'Torsion constant of the beam: {J:.2e} m4')
print(f'Shear modulus of the beam: {G:.2e} Pa')
print(f'Density of the beam: {rho_c:.2e} kg/m3')
print()

Beam_m = 150000                    # [kg/m]
Beam_EI = 5*E_c*(np.pi*a**3*b)/4      # [N.m2]
Beam_EIz = E_c*(np.pi*a*b**3)/4
Beam_EA = E_c*A                    # [N]
Beam_GJ = G*J                      # [N.m2]
Beam_Im = rho_c*J                  #[m2]
print(f'Mass per unit length of the beam: {Beam_m:.2e} kg/m')
print(f'Bending stiffness of the beam, x-direction : {Beam_EI:.2e} N.m2')
print(f'Bending stiffness of the beam, z-direction: {Beam_EIz:.2e} N.m2')
print(f'Axial stiffness of the beam: {Beam_EA:.2e} N')
print(f'Torsional stiffness of the beam: {Beam_GJ:.2e} N.m2')
print(f'Moment of inertia of the beam: {Beam_Im:.2e} m2')


# ----------------------- Creating Tunnel elements ----------------------- #
# Length scales
DistLandings = 4000 # Distance between landings, distance in x-direction
Deviation = 2000 # The maximum deviation, distance in y-direction

r = sp.symbols('r')
# Equation for the tunnel
equation = sp.Eq((DistLandings/2)**2 + (r - Deviation)**2, r**2)
solutions = sp.solve(equation, r)
#TunRad = float(solutions[0]) #m

#TunAng = (180-np.arccos((DistLandings/2)/TunRad)*180/np.pi,
          #np.arccos((DistLandings/2)/TunRad)*180/np.pi) #deg

# dth in degrees (angle increment)
#dth = 0.0509 #deg

TunRad = 20050 #m
TunAng = (95.725, 84.275) #deg
steps = 10
dth = (TunAng[0] - TunAng[1]) / steps #deg


TunCX = TunRad*np.cos(np.deg2rad(
    np.arange(TunAng[0], TunAng[1]-dth, -dth) ) )+ DistLandings/2
TunCY = TunRad*np.sin(np.deg2rad(np.arange(TunAng[0],
                                           TunAng[1]-dth,
                                           -dth) ) )
TunCY = TunCY - min(TunCY)
TunCZ = np.zeros(len(TunCX))  # z-coordinates are zero


# ----------------------- Creating the nodes ----------------------- #
NodeC = [ [x,y] for x,y in zip(TunCX, TunCY) ]
nNode = len(NodeC)


# Define elements (and their properties
#             NodeLeft    NodeRight          m         EA        EI
Ele = [ [n1, n2, Beam_m, Beam_EA, Beam_EI] 
           for n1,n2 in zip(range(0,nNode-1), range(1,nNode)) ]
nEle = len(Ele)


# ----------------------- Creating the matrix ----------------------- #
LDOF = 3
nDof = LDOF*nNode  #3 dof per node
K = np.zeros(nDof*nDof)
M = np.zeros(nDof*nDof)
Q = np.zeros(nDof)
    

exeTime = [0.0, 0.0]
exeTime[0] = time.time()

for iEle in range(0, nEle):
    n1, n2, m, EA, EI = Ele[iEle]
    
    x1, y1 = NodeC[n1]
    x2, y2 = NodeC[n2]
    
    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    
    n1dof = LDOF*n1 + np.arange(0,LDOF)
    n2dof = LDOF*n2 + np.arange(0,LDOF)    
    
    # Calculate elemental matrices
    Me, Ke, alpha, T_transformation = Beam2DMatrices(m, EA, EI,
                                                     (NodeC[n1], NodeC[n2]) )
    
    
    indexes = np.append(n1dof, n2dof)
    for i in range(0, 2*LDOF):
        for j in range(0, 2*LDOF):
            ij = indexes[i]*nDof + indexes[j]
            M[ij] = M[ij] + Me[i,j]
            K[ij] = K[ij] + Ke[i,j]
    

# Reshape the global matrix from a 1-dimensional array
# to a 2-dimensional array
M = M.reshape((nDof, nDof))
K = K.reshape((nDof, nDof))

# ----------------------- Apply boundary conditions ----------------------- #
NodesClamp = (0, nNode-1)

# Prescribed dofs
DofsP = np.empty([0], dtype=int)
for n0 in NodesClamp:
    DofsP = np.append(DofsP, n0*LDOF + np.arange(0,LDOF))

# Free dofs
DofsF = np.arange(0, nDof)       # free DOFs
DofsF = np.delete(DofsF, DofsP)  # remove the fixed DOFs from 
                                 # the free DOFs array


M_FF = [ M[iRow,DofsF].tolist() for iRow in DofsF ]
K_FF = [ K[iRow,DofsF].tolist() for iRow in DofsF ]


# ----------------------- Redoing some notation stuff ----------------------- #
# free & fixed array indices
fx = DofsF[:, np.newaxis]
fy = DofsF[np.newaxis, :]
bx = DofsP[:, np.newaxis]
by = DofsP[np.newaxis, :]

# Mass
Mii = M[fx, fy]
Mib = M[fx, by]
Mbi = M[bx, fy]
Mbb = M[bx, by]

# Stiffness
Kii = K[fx, fy]
Kib = K[fx, by]
Kbi = K[bx, fy]
Kbb = K[bx, by]

# External force
Qi = Q[fx]
Qb = Q[bx]


# Rewriting some notation 
M_FP = Mib
K_FP = Kib



ax = plt.axes(projection='3d')

# add dotted black lines to indicate the max tunnel extent
coord_start = [TunCX[0], TunCY[0], TunCZ[0]]
coord_end = [TunCX[-1], TunCY[-1], TunCZ[-1]]
coord_start_min_y = [TunCX[0], max(TunCY), TunCZ[0]]
coord_end_max_y = [TunCX[-1], max(TunCY), TunCZ[-1]]
plt.plot([coord_start[0], coord_end[0],
          coord_end_max_y[0], coord_start_min_y[0],
          coord_start[0]],
         [coord_start[1], coord_end[1],
          coord_end_max_y[1], coord_start_min_y[1],
          coord_start[1]],
         [coord_start[2], coord_end[2],
          coord_end_max_y[2], coord_start_min_y[2],
          coord_start[2]], 'k--')

# ax.view_init(45, 30)
plt.plot(TunCX, TunCY, TunCZ, lw=5)
plt.grid("on")
plt.axis("equal")
plt.xlabel('x (m)')
plt.ylabel('y (m)')
ax.set_zlabel('z (m)')  # Add label for the 3rd axis

plt.xlim([0, max(TunCX)])
plt.ylim([-5000, 5000])
ax.set_zlim([-5000, 5000])

plt.title(f'Tunnel Geometry (deviation in Y: '
          + f'{max(TunCY)-min(TunCY):.0f} m,'
          + f'Z: {max(TunCZ)-min(TunCZ):.0f} m)')

print()
print(f'Number of tunnel elements: {K.shape[0]/6}') # 6 dof per element
print('Deviation of tunnel alignment from straight line, in:')
print(f'  X: {max(TunCX)-min(TunCX):g} m')
print(f'  Y: {max(TunCY)-min(TunCY):g} m')
print(f'  Z: {max(TunCZ)-min(TunCZ):g} m')

# # Look at the matrix structure
# plt.figure()
# plt.spy(M_FP)
# plt.title("Mass matrix")

# Return stiffness vector
def return_stiffness_vector():
    return K_FF


# Calculate force matrix
def Force_vector(total_force = 100):
    Q = np.zeros(nDof)
    for i in range(0, nNode):
        for k in range(0, LDOF):
            if i == 0 or i == nNode-1:
                Q[LDOF*i + k] = 0.5*total_force*17000/nEle
            elif k == 1:
                Q[LDOF*i + k] = total_force*17000/nEle
                
    # Construct a matrix to obtain the correct entities of the Q vector
    R1 = np.identity(nDof)
    R1 = R1[fx, 0:nDof]
    F_external = np.dot(R1, Q).reshape(1,-1)
    # Get rid of matrix form and transform this obtained matrix back
    # into a vector
    F_external = F_external[0,:]
    return F_external

#catenary_end


# Calculate forces in mooring
def calculate_axial_stress_FEM(Total_force):    
    u_displ_pos = scipy.linalg.solve(K_FF, Force_vector(Total_force))
    max_defl = np.max(u_displ_pos)
    #print('max', max_defl)
    u_displ_neg = scipy.linalg.solve(K_FF, Force_vector(-Total_force))
    min_defl = np.max(u_displ_neg)
    #print('min', min_defl)

    L_0 = 400 #m
    delta_l = (max_defl - min_defl)/2
    print(delta_l)
    eps = delta_l / L_0
    E = 210000 #N/mm^2
    

    return E * eps
    