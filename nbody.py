import numpy as np
import time
import sys
from numba import njit, prange

# --- 1. Pure Python (Very Slow, for baseline) ---
def update_physics_pure(pos, vel, mass, dt, steps):
    N = len(pos)
    for _ in range(steps):
        for i in range(N):
            Fx, Fy, Fz = 0.0, 0.0, 0.0
            for j in range(N):
                if i == j: continue
                dx = pos[j,0] - pos[i,0]
                dy = pos[j,1] - pos[i,1]
                dz = pos[j,2] - pos[i,2]
                dist_sq = dx*dx + dy*dy + dz*dz + 1e-9
                dist = dist_sq ** 0.5
                f = (mass[i] * mass[j]) / (dist_sq * dist)
                Fx += f * dx
                Fy += f * dy
                Fz += f * dz
            vel[i,0] += Fx * dt / mass[i]
            vel[i,1] += Fy * dt / mass[i]
            vel[i,2] += Fz * dt / mass[i]
        pos += vel * dt

# --- 2. Numba Parallel (Uses OpenMP backend) ---
@njit(parallel=True, fastmath=True)
def update_physics_numba(pos, vel, mass, dt, steps):
    N = len(pos)
    for _ in range(steps):
        # prange explicitly tells Numba to parallelize this loop (like #pragma omp parallel for)
        for i in prange(N):
            Fx, Fy, Fz = 0.0, 0.0, 0.0
            for j in range(N):
                if i == j: continue
                dx = pos[j,0] - pos[i,0]
                dy = pos[j,1] - pos[i,1]
                dz = pos[j,2] - pos[i,2]
                dist_sq = dx*dx + dy*dy + dz*dz + 1e-9
                dist = dist_sq ** 0.5
                f = (mass[i] * mass[j]) / (dist_sq * dist)
                Fx += f * dx
                Fy += f * dy
                Fz += f * dz
            vel[i,0] += Fx * dt / mass[i]
            vel[i,1] += Fy * dt / mass[i]
            vel[i,2] += Fz * dt / mass[i]
        
        # Parallelize position update too
        for i in prange(N):
            pos[i,0] += vel[i,0] * dt
            pos[i,1] += vel[i,1] * dt
            pos[i,2] += vel[i,2] * dt

if __name__ == "__main__":
    mode = sys.argv[1] # "pure" or "numba"
    N = int(sys.argv[2])
    steps = int(sys.argv[3])
    
    pos = np.random.rand(N, 3).astype(np.float32)
    vel = np.zeros((N, 3), dtype=np.float32)
    mass = np.ones(N, dtype=np.float32)
    
    start = time.time()
    if mode == "pure":
        update_physics_pure(pos, vel, mass, 0.01, steps)
    elif mode == "numba":
        # First call includes compilation time, so we run a dummy first
        update_physics_numba(pos, vel, mass, 0.01, 1) 
        start = time.time() # Reset timer
        update_physics_numba(pos, vel, mass, 0.01, steps)
        
    print(time.time() - start)