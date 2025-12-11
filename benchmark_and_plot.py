import subprocess
import matplotlib.pyplot as plt
import sys
import os

# --- Configuration ---
body_counts = [100, 500, 1000, 2000] 
steps = 20

# Executables
executables = {
    "C Serial": "./nbody_c_serial",
    "C OpenMP": "./nbody_c_omp",
    "C++ Serial": "./nbody_cpp_serial",
    "C++ OpenMP": "./nbody_cpp_omp"
}

results = {
    "C Serial": [], "C OpenMP": [],
    "C++ Serial": [], "C++ OpenMP": [],
    "Python Pure": [], "Python Numba": []
}

def run_command(cmd, label, N):
    try:
        # Run command and capture output
        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
        time_taken = float(output.strip())
        return time_taken
    except subprocess.CalledProcessError as e:
        print(f"\nError running {label} with N={N}:")
        print(e.output.decode())
        return None
    except ValueError:
        return None

if __name__ == "__main__":
    print(f"{'='*60}")
    print(f"Starting N-Body Benchmark")
    print(f"Python Interpreter: {sys.executable}") # This confirms which python is running
    print(f"{'='*60}\n")

    print(f"{'Type':<15} | {'N':<6} | {'Time (s)':<10}")
    print("-" * 35)

    for N in body_counts:
        # 1. Run C/C++ Executables
        for label, exe_path in executables.items():
            if os.path.exists(exe_path):
                cmd = f"{exe_path} {N} {steps}"
                t = run_command(cmd, label, N)
                results[label].append(t)
                if t: print(f"{label:<15} | {N:<6} | {t:.5f}")
            else:
                results[label].append(None)

        # 2. Run Python (Pure)
        # Note: We use 'nbody.py' here because your logs show that is your filename
        if N <= 500:
            # sys.executable forces the script to use the CURRENT python (Anaconda)
            cmd = f"{sys.executable} nbody.py pure {N} {steps}"
            t = run_command(cmd, "Python Pure", N)
            results["Python Pure"].append(t)
            if t: print(f"{'Python Pure':<15} | {N:<6} | {t:.5f}")
        else:
            results["Python Pure"].append(None)
            print(f"{'Python Pure':<15} | {N:<6} | SKIPPED")

        # 3. Run Python (Numba)
        cmd = f"{sys.executable} nbody.py numba {N} {steps}"
        t = run_command(cmd, "Python Numba", N)
        results["Python Numba"].append(t)
        if t: print(f"{'Python Numba':<15} | {N:<6} | {t:.5f}")
        
        print("-" * 35)

    # --- Plotting ---
    print("\nGenerating plot...")
    plt.figure(figsize=(12, 8))

    styles = {
        "C Serial":     {'color': 'red', 'linestyle': '--', 'marker': 'o'},
        "C OpenMP":     {'color': 'red', 'linestyle': '-',  'marker': 's', 'linewidth': 2},
        "C++ Serial":   {'color': 'blue', 'linestyle': '--', 'marker': 'o'},
        "C++ OpenMP":   {'color': 'blue', 'linestyle': '-',  'marker': 's', 'linewidth': 2},
        "Python Pure":  {'color': 'green', 'linestyle': ':', 'marker': 'x'},
        "Python Numba": {'color': 'orange', 'linestyle': '-', 'marker': '^', 'linewidth': 2}
    }

    for label, times in results.items():
        valid_data = [(n, t) for n, t in zip(body_counts, times) if t is not None]
        if valid_data:
            x, y = zip(*valid_data)
            plt.plot(x, y, label=label, **styles.get(label, {}))

    plt.xlabel('Number of Bodies (N)')
    plt.ylabel('Execution Time (seconds) [Log Scale]')
    plt.title(f'N-Body Simulation: Serial vs Parallel ({steps} steps)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.yscale('log')
    
    plt.savefig('nbody_benchmark_results.png')
    print(f"Done! Plot saved to 'nbody_benchmark_results.png'")