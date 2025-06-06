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


def run_simulation(pdb_id, gpu_index, steps, q):
    print(f"Fetching PDB {pdb_id}...")
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

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
    PDBFile.writeFile(fixer.topology, fixer.positions, open(f"outputs/{pdb_id}_GPU{gpu_index}_solvated.pdb", 'w'))

    # Count atoms after solvation
    atom_count = len(list(fixer.topology.atoms()))

    # Create OpenMM system
    forcefield = ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    system = forcefield.createSystem(fixer.topology, 
                                     nonbondedMethod=PME,
                                     nonbondedCutoff=1*nanometers,
                                     constraints=HBonds)

    integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)

    # Use the GPU platform
    platform = Platform.getPlatformByName('CUDA')
    properties = {'CudaDeviceIndex': str(gpu_index), 'CudaPrecision': 'mixed'}  # for speed/accuracy balance

    simulation = Simulation(fixer.topology, system, integrator, platform, properties)
    simulation.context.setPositions(fixer.positions)

    print("Minimizing energy...")
    simulation.minimizeEnergy()

    # Save coordinates after minimization
    state = simulation.context.getState(getPositions=True)
    PDBFile.writeFile(simulation.topology, state.getPositions(), open(f"outputs/{pdb_id}_GPU{gpu_index}_minimized.pdb", 'w'))

    print("Running simulation...")
    simulation.context.setVelocitiesToTemperature(300*kelvin)
    simulation.reporters.append(StateDataReporter(sys.stdout, 1000, step=True, potentialEnergy=True, temperature=True, speed=True))

    # Start GPU power monitoring
    global monitoring
    monitoring = True
    monitor_thread = threading.Thread(target=monitor_gpu_power, args=(0.5,), daemon=True)
    monitor_thread.start()

    start = time.time()
    simulation.step(steps)
    end = time.time()

    monitoring = False
    monitor_thread.join()

    # Save final coordinates
    final_state = simulation.context.getState(getPositions=True)
    PDBFile.writeFile(simulation.topology, final_state.getPositions(), open(f"outputs/{pdb_id}_GPU{gpu_index}_final.pdb", 'w'))

    # Process results
    result = f'PDB: {pdb_id}, GPU: {gpu_index}, Steps: {steps}, Timestep: {0.002*picoseconds}, Residues: {residue_count}, Atoms: {atom_count}\n'
    gpu_power_log_clean = [r[1] for r in gpu_power_log if all(isinstance(p, float) for p in r[1])]
    if gpu_power_log_clean:
        avg_power = [statistics.mean(p[i] for p in gpu_power_log_clean) for i in range(len(gpu_power_log_clean[0]))]
        max_power = [max(p[i] for p in gpu_power_log_clean) for i in range(len(gpu_power_log_clean[0]))]
        #print("\nGPU Average Power Usage (Watts):")
        #result += f"\nGPU {gpu_index} Average Power Usage (Watts):\n"
        for i, power in enumerate(avg_power):
            if str(i) == gpu_index:
                #print(f"GPU {i}: {power:.2f} W, Max = {max_power[i]:.2f} W")
                result += f"STATS: {pdb_id}, GPU {i}: Mean = {power:.2f} W, Max = {max_power[i]:.2f} W\n"
    else:
        #print("No valid GPU power data recorded.")
        result += "No valid GPU power data recorded.\n"

    ns_per_day = (steps * 0.002 * 86400) / (end - start)
    result += f"Time taken: {end - start:.2f} s\n"
    #result += f"Performance: {ns_per_day:.2f} ns/day\n"
    q.put(result)
    #print(f"Benchmark complete for {pdb_id}")
    #print(f"Time taken: {end - start:.2f} s")
    #print(f"Performance: {ns_per_day:.2f} ns/day")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_id", help="4-letter PDB ID (e.g. 1UBQ)")
    parser.add_argument("--steps", type=int, default=10000, help="Number of MD steps")
    args = parser.parse_args()

    # Run the simulation
    #run_simulation(args.pdb_id.upper(), args.steps)
    q = Queue()
    sim1 = Process(target=run_simulation, args=(args.pdb_id.upper(), '0', args.steps, q))
    sim2 = Process(target=run_simulation, args=(args.pdb_id.upper(), '1', args.steps, q))

    sim1.start()
    sim2.start()

    sim1.join()
    sim2.join()

    print("Both simulations completed.")

    print("\n=== Benchmark Results ===")
    while not q.empty():
        print(q.get())
