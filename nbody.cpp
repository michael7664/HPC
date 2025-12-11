#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

#ifdef _OPENMP
#include <omp.h>
#else
#include <ctime>
#include <sys/time.h>
#endif

struct Body {
    float x, y, z;
    float vx, vy, vz;
    float mass;
};

double get_time() {
    #ifdef _OPENMP
        return omp_get_wtime();
    #else
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return tv.tv_sec + tv.tv_usec * 1e-6;
    #endif
}

void update_physics(std::vector<Body>& bodies, float dt, int steps) {
    int N = bodies.size();
    for (int t = 0; t < steps; t++) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) {
            float Fx = 0, Fy = 0, Fz = 0;
            for (int j = 0; j < N; j++) {
                if (i == j) continue;
                float dx = bodies[j].x - bodies[i].x;
                float dy = bodies[j].y - bodies[i].y;
                float dz = bodies[j].z - bodies[i].z;
                float dist_sq = dx*dx + dy*dy + dz*dz + 1e-9f;
                float dist = std::sqrt(dist_sq);
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
    if (argc != 3) return 1;
    int N = std::atoi(argv[1]);
    int steps = std::atoi(argv[2]);
    float dt = 0.01f;

    std::vector<Body> bodies(N);
    for (int i = 0; i < N; i++) {
        bodies[i] = {
            (float)(rand() % 100), (float)(rand() % 100), (float)(rand() % 100),
            0, 0, 0, 1.0f
        };
    }

    double start = get_time();
    update_physics(bodies, dt, steps);
    double end = get_time();

    std::cout << (end - start);
    return 0;
}