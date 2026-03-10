import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import linear_sum_assignment

# =========================
# Parameters
# =========================
n = 6
dim = 2
epsilon = 0.01
nu = 1
beta = 0.5
L = 6

M1 = -10*np.eye(dim)
M2 = -10*np.eye(dim)
M3 = -10*np.eye(dim)
M4 = -10*np.eye(dim)

# =========================
# FORMATIONS
# =========================

# Hexagon
P1 = np.zeros((n,dim))
for i in range(n):
    P1[i,:] = L*np.array([np.cos((i+1)*np.pi/3), np.sin((i+1)*np.pi/3)], dtype=float)

# Vertical line
y_vals = np.linspace(-2.5*L,2.5*L,n)
P2 = np.column_stack((np.zeros(n),y_vals))

# V formation
P3 = np.array([
    [2,0],
    [0,2],
    [0,-2],
    [-2,4],
    [-2,-4],
    [-2,0]
], dtype=float)

# Grid
X_grid = np.array([-1,0,1,-1,0,1])*1.5
Y_grid = np.array([1,1,1,-1,-1,-1])*1.5
P4 = np.column_stack((X_grid,Y_grid))

# Center formations
P1 -= np.mean(P1,axis=0)
P2 -= np.mean(P2,axis=0)
P3 -= np.mean(P3,axis=0)
P4 -= np.mean(P4,axis=0)

# Initial positions with fixed seed 
np.random.seed(42)
Z0 = 5*np.random.randn(dim*n)

# =========================
# Reference trajectory
# =========================
def r(t):
    return np.array([t,0])

# =========================
# Reassignment Logic
# =========================
def reassign_formation(Z_curr_flat, P_new, t_curr, n, dim):
    """
    Reorders the rows of P_new to minimize the total travel distance 
    from the agents' current physical positions.
    """
    # 1. Get current physical positions of agents
    Z_curr = Z_curr_flat.reshape((n, dim))
    
    # 2. Get the physical world positions of the new target slots at this exact time
    ref_pos = r(t_curr)
    target_world = P_new + ref_pos
    
    # 3. Build a cost matrix (squared Euclidean distance)
    # cost_matrix[i, j] is the distance from agent i to target slot j
    diff = Z_curr[:, None, :] - target_world[None, :, :]
    cost_matrix = np.sum(diff**2, axis=2)
    
    # 4. Solve the Linear Sum Assignment problem (Hungarian Algorithm)
    # row_ind is the agent index, col_ind is the assigned target slot index
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # 5. Re-order the P matrix so agent 'i' gets the slot 'col_ind[i]'
    P_reordered = P_new[col_ind]
    
    return P_reordered

# =========================
# Dynamics
# =========================
def dynamics(t, Z, n, dim, M, epsilon, nu, beta, P):

    Zmat = Z.reshape((n, dim))

    # centroid
    c = np.mean(Zmat, axis=0)

    # reference trajectory
    r_val = np.array([t,0])
    r_dot = np.array([1,0])

    sigma = c - r_val
    zeta = np.sign(sigma)*(np.abs(sigma)**beta)

    # ==================================
    # Vectorized pairwise differences
    # ==================================

    # biased positions
    B = Zmat - P

    # pairwise differences
    diff = B[:,None,:] - B[None,:,:]      # shape (n,n,dim)

    # norms
    norms = np.linalg.norm(diff, axis=2)

    # avoid division by zero
    denom = norms**nu + epsilon

    # apply interaction matrix
    Md = diff @ M.T

    # divide by denominator
    interaction = Md / denom[:,:,None]

    # remove self interaction
    np.fill_diagonal(interaction[:,:,0],0)
    np.fill_diagonal(interaction[:,:,1],0)

    # sum over neighbors
    sum_term = np.sum(interaction, axis=1)/n

    # dynamics equation
    dZ = -Zmat - sum_term + r_val + r_dot - zeta

    return dZ.flatten()

# =========================
# Time intervals & Simulation
# =========================
t1 = (0,25)
t2 = (25,50)
t3 = (50,75)
t4 = (75,100)

print("Simulating dynamics...")

# Stage 1: Random start, so P1 doesn't need reassignment
sol1 = solve_ivp(dynamics, t1, Z0, args=(n, dim, M1, epsilon, nu, beta, P1))

# Stage 2: Reassign P2 based on end of Stage 1
Z1_end = sol1.y[:, -1]
P2_opt = reassign_formation(Z1_end, P2, t2[0], n, dim)
sol2 = solve_ivp(dynamics, t2, Z1_end, args=(n, dim, M2, epsilon, nu, beta, P2_opt))

# Stage 3: Reassign P3 based on end of Stage 2
Z2_end = sol2.y[:, -1]
P3_opt = reassign_formation(Z2_end, P3, t3[0], n, dim)
sol3 = solve_ivp(dynamics, t3, Z2_end, args=(n, dim, M3, epsilon, nu, beta, P3_opt))

# Stage 4: Reassign P4 based on end of Stage 3
Z3_end = sol3.y[:, -1]
P4_opt = reassign_formation(Z3_end, P4, t4[0], n, dim)
sol4 = solve_ivp(dynamics, t4, Z3_end, args=(n, dim, M4, epsilon, nu, beta, P4_opt))

# Combine the data for plotting
t = np.concatenate((sol1.t, sol2.t, sol3.t, sol4.t))
Z = np.hstack((sol1.y, sol2.y, sol3.y, sol4.y)).T

print("Simulation complete")

# =========================
# Animation
# =========================

fig,ax = plt.subplots(figsize=(10,4))

ax.set_xlim(-10,110)
ax.set_ylim(-15,15)
ax.set_aspect('equal')
ax.grid(True)

ax.set_title("Swarm Formation Morphing")
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")

# reference path
r_traj = np.array([r(ti) for ti in t])
ax.plot(r_traj[:,0],r_traj[:,1],'r--',label="Reference Path")

colors = plt.cm.tab10(np.linspace(0,1,n))

agents=[]
trails=[]

for i in range(n):

    trail, = ax.plot([],[],lw=1,color=colors[i],alpha=0.4)
    agent, = ax.plot([],[],'o',color=colors[i],markeredgecolor='k')

    trails.append(trail)
    agents.append(agent)

target, = ax.plot([],[],'ks',markersize=8,markerfacecolor='r',label="Target")

ax.legend()

speed = max(1.0,len(t)//500)

for k in range(0,len(t),speed):

    target.set_data(r_traj[k,0],r_traj[k,1])

    for i in range(n):

        x = Z[k,(i*dim)]
        y = Z[k,(i*dim)+1]

        agents[i].set_data(x,y)

        trails[i].set_data(
            Z[:k,(i*dim)],
            Z[:k,(i*dim)+1]
        )

    plt.pause(0.01)

plt.show()