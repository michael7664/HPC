#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Conditional Include for OpenMP
#ifdef _OPENMP
#include <omp.h>
#else
#include <time.h>
#include <sys/time.h>
#endif

typedef struct {
    float x, y, z;
    float vx, vy, vz;
    float mass;
} Body;

// Cross-platform timer function
double get_time() {
    #ifdef _OPENMP
        return omp_get_wtime();
    #else
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return tv.tv_sec + tv.tv_usec * 1e-6;
    #endif
}

void update_physics(Body* bodies, int N, float dt, int steps) {
    for (int t = 0; t < steps; t++) {
        // Only parallelize if OpenMP is active
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) {
            float Fx = 0, Fy = 0, Fz = 0;
            for (int j = 0; j < N; j++) {
                if (i == j) continue;
                float dx = bodies[j].x - bodies[i].x;
                float dy = bodies[j].y - bodies[i].y;
                float dz = bodies[j].z - bodies[i].z;
                float dist_sq = dx*dx + dy*dy + dz*dz + 1e-9f;
                float dist = sqrtf(dist_sq);
                float f = (1.0f * bodies[i].mass * bodies[j].mass) / (dist_sq * dist);
                Fx += f * dx;
                Fy += f * dy;
                Fz += f * dz;
            }
            bodies[i].vx += Fx * dt / bodies[i].mass;
            bodies[i].vy += Fy * dt / bodies[i].mass;
            bodies[i].vz += Fz * dt / bodies[i].mass;
        }

        for (int i = 0; i < N; i++) {
            bodies[i].x += bodies[i].vx * dt;
            bodies[i].y += bodies[i].vy * dt;
            bodies[i].z += bodies[i].vz * dt;
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 3) { printf("Usage: %s <N> <steps>\n", argv[0]); return 1; }
    int N = atoi(argv[1]);
    int steps = atoi(argv[2]);
    float dt = 0.01f;

    Body* bodies = (Body*)malloc(N * sizeof(Body));
    
    for (int i = 0; i < N; i++) {
        bodies[i].x = (float)(rand() % 100);
        bodies[i].y = (float)(rand() % 100);
        bodies[i].z = (float)(rand() % 100);
        bodies[i].vx = 0; bodies[i].vy = 0; bodies[i].vz = 0;
        bodies[i].mass = 1.0f;
    }

    double start = get_time();
    update_physics(bodies, N, dt, steps);
    double end = get_time();

    printf("%f", end - start);
    free(bodies);
    return 0;
}