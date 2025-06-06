from openmm.app import *
from openmm import *
from openmm.unit import *
from pdbfixer import PDBFixer
import argparse
import time
import sys
import threading
import pynvml
import statistics
import numpy as np
from multiprocessing import Process, Queue
import os
import sys
import time
import threading
import statistics
from pdbfixer import PDBFixer
from openmm.app import *
from openmm import *
from openmm.unit import *
import numpy as np

###################################################################
gpu_power_log = []
monitoring = True
def monitor_gpu_power(interval=1.0):
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    while monitoring:
        readings = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # milliwatts to watts
            except pynvml.NVMLError:
                power = float('nan')
            readings.append(power)
        gpu_power_log.append((time.time(), readings))
        time.sleep(interval)
    pynvml.nvmlShutdown()

###################################################################
cpu_power_log = []
def monitor_cpu_power(interval=0.5):
    global monitoring
    print("Starting CPU power monitoring...")
    
    energy_path = "/sys/class/powercap/intel-rapl:0/energy_uj"
    has_powercap = os.path.exists(energy_path)
    
    try:
        prev_energy = None
        prev_time = None

        while monitoring:
            timestamp = time.time()
            usage = psutil.cpu_percent(interval=None)

            if has_powercap:
                with open(energy_path, 'r') as f:
                    energy = int(f.read().strip())
                if prev_energy is not None:
                    delta_energy = energy - prev_energy
                    delta_time = timestamp - prev_time
                    power_watts = (delta_energy / 1e6) / delta_time  # convert from µJ to W
                    cpu_power_log.append((timestamp, usage, power_watts))
                prev_energy = energy
                prev_time = timestamp
            else:
                cpu_power_log.append((timestamp, usage, None))

            time.sleep(interval)
    except Exception as e:
        print(f"CPU power monitoring failed: {e}")

###################################################################
def run_simulation(pdb_id, steps, q, use_gpu, gpu_index, cpu_index):
    print(f"Fetching PDB {pdb_id}...")

    if not use_gpu and cpu_index is not None:
        try:
            os.sched_setaffinity(0, {cpu_index})
            print(f"Process affinity set to CPU core {cpu_index}")
        except AttributeError:
            print("os.sched_setaffinity not supported on this platform.")
        except Exception as e:
            print(f"Failed to set CPU affinity: {e}")


    # Load and fix the PDB
    fixer = PDBFixer(pdbid=pdb_id)
    fixer.removeHeterogens(True)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=7.0)

    # Count residues before solvation
    residue_count = len(list(fixer.topology.residues()))
    
    # Solvate in a cubic water box
    print("Adding solvent...")
    fixer.addSolvent(padding=1.0*nanometers)

    # Save coordinates after solvation
    os.makedirs("outputs", exist_ok=True)
    if use_gpu:
        PDBFile.writeFile(fixer.topology, fixer.positions, open(f"outputs/{pdb_id}_GPU{gpu_index}_solvated.pdb", 'w'))
    elif not use_gpu:
        PDBFile.writeFile(fixer.topology, fixer.positions, open(f"outputs/{pdb_id}_CPU{cpu_index}_solvated.pdb", 'w'))


    # Count atoms after solvation
    atom_count = len(list(fixer.topology.atoms()))

    # Create OpenMM system
    forcefield = ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    system = forcefield.createSystem(fixer.topology, 
                                     nonbondedMethod=PME,
                                     nonbondedCutoff=1*nanometers,
                                     constraints=HBonds)

    integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)

    # Select platform
    if use_gpu:
        platform = Platform.getPlatformByName('CUDA')
        properties = {'CudaDeviceIndex': str(gpu_index), 'CudaPrecision': 'mixed'}
    else:
        platform = Platform.getPlatformByName('CPU')
        properties = {}

    simulation = Simulation(fixer.topology, system, integrator, platform, properties)
    simulation.context.setPositions(fixer.positions)

    print("Minimizing energy...")
    simulation.minimizeEnergy()

    # Save coordinates after minimization
    state = simulation.context.getState(getPositions=True)
    if use_gpu:
        PDBFile.writeFile(simulation.topology, state.getPositions(), open(f"outputs/{pdb_id}_GPU{gpu_index}_minimized.pdb", 'w'))
    elif not use_gpu:
        PDBFile.writeFile(simulation.topology, state.getPositions(), open(f"outputs/{pdb_id}_CPU{cpu_index}_minimized.pdb", 'w'))

    print("Running simulation...")
    simulation.context.setVelocitiesToTemperature(300*kelvin)
    simulation.reporters.append(StateDataReporter(sys.stdout, 1000, step=True, potentialEnergy=True, temperature=True, speed=True))

    # GPU power monitoring (only if using GPU)
    global monitoring
    if use_gpu:
        monitoring = True
        monitor_thread = threading.Thread(target=monitor_gpu_power, args=(0.5,), daemon=True)
        monitor_thread.start()

    # CPU power monitoring (only if not using GPU)
    if not use_gpu:
        monitoring = True
        monitor_thread = threading.Thread(target=monitor_cpu_power, args=(0.5,), daemon=True)
        monitor_thread.start()

    start = time.time()
    simulation.step(steps)
    end = time.time()

    if use_gpu:
        monitoring = False
        monitor_thread.join()

    if not use_gpu:
        monitoring = False
        monitor_thread.join()

    # Save final coordinates
    final_state = simulation.context.getState(getPositions=True)
    if use_gpu:
        PDBFile.writeFile(simulation.topology, final_state.getPositions(), open(f"outputs/{pdb_id}_GPU{gpu_index}_final.pdb", 'w'))
        result = f'PDB: {pdb_id}, GPU: {gpu_index if use_gpu else "CPU"}, Steps: {steps}, Timestep: {0.002*picoseconds}, Residues: {residue_count}, Atoms: {atom_count}\n'

        if use_gpu and 'gpu_power_log' in globals():
            gpu_power_log_clean = [r[1] for r in gpu_power_log if all(isinstance(p, float) for p in r[1])]
            if gpu_power_log_clean:
                avg_power = [statistics.mean(p[i] for p in gpu_power_log_clean) for i in range(len(gpu_power_log_clean[0]))]
                max_power = [max(p[i] for p in gpu_power_log_clean) for i in range(len(gpu_power_log_clean[0]))]
                for i, power in enumerate(avg_power):
                    if str(i) == str(gpu_index):
                        result += f"STATS: {pdb_id}, GPU {i}: Mean = {power:.2f} W, Max = {max_power[i]:.2f} W\n"
            else:
                result += "No valid GPU power data recorded.\n"

    elif not use_gpu:
        PDBFile.writeFile(simulation.topology, final_state.getPositions(), open(f"outputs/{pdb_id}_CPU{cpu_index}_final.pdb", 'w'))
        result = f'PDB: {pdb_id}, CPU: {cpu_index if use_gpu else "CPU"}, Steps: {steps}, Timestep: {0.002*picoseconds}, Residues: {residue_count}, Atoms: {atom_count}\n'

        if not use_gpu and 'cpu_power_log' in globals():
            cpu_power_log_clean = [r[1] for r in gpu_power_log if all(isinstance(p, float) for p in r[1])]
            if cpu_power_log_clean:
                avg_power = [statistics.mean(p[i] for p in cpu_power_log_clean) for i in range(len(cpu_power_log_clean[0]))]
                max_power = [max(p[i] for p in cpu_power_log_clean) for i in range(len(cpu_power_log_clean[0]))]
                for i, power in enumerate(avg_power):
                    if str(i) == str(cpu_index):
                        result += f"STATS: {pdb_id}, CPU {i}: Mean = {power:.2f} W, Max = {max_power[i]:.2f} W\n"
            else:
                result += "No valid GPU power data recorded.\n"

    ns_per_day = (steps * 0.002 * 86400) / (end - start)
    result += f"Time taken: {end - start:.2f} s\n"

    q.put(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_id", help="4-letter PDB ID (e.g. 1UBQ)")
    parser.add_argument("--steps", type=int, default=10000, help="Number of MD steps")
    args = parser.parse_args()

    # Run the simulation
    #run_simulation(args.pdb_id.upper(), args.steps)
    q = Queue()
    sim1 = Process(target=run_simulation, args=(args.pdb_id.upper(), args.steps, q, True, '0', None))
    sim2 = Process(target=run_simulation, args=(args.pdb_id.upper(), args.steps, q, True, '1', None))
    cpu_sims = [Process(target=run_simulation, args=(args.pdb_id.upper(), args.steps, q, False, None, i)) for i in range(0, 50)]


    sim1.start()
    time.sleep(5)
    sim2.start()
    for sim in cpu_sims:
        time.sleep(5)
        sim.start()


    sim1.join()
    time.sleep(5)
    sim2.join()
    for sim in cpu_sims:
        time.sleep(5)
        sim.join()

    print("Both simulations completed.")

    print("\n=== Benchmark Results ===")
    while not q.empty():
        print(q.get())
    print('NORMAL TERMINATION')
