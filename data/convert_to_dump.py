import numpy as np

def npy_to_lammps_dump(npy_file, output_dump):
    # 1. Load the extracted tensor
    print(f"Loading tensor from {npy_file}...")
    tensor = np.load(npy_file)
    T, N, _ = tensor.shape
    
    print(f"Loaded tensor with {T} frames and {N} atoms.")
    print(f"Writing to {output_dump}...")
    
    # 2. Write the formatted LAMMPS dump file
    with open(output_dump, 'w') as f:
        for t in range(T):
            # Write timestep (Assuming the original 10-step intervals)
            f.write("ITEM: TIMESTEP\n")
            f.write(f"{t * 10}\n")
            
            # Write number of atoms
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(f"{N}\n")
            
            # Write box bounds (Using the bounds from the initial PMMA_data_initial.data file)
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            f.write("-100.0 100.0\n")
            f.write("-100.0 100.0\n")
            f.write("-100.0 100.0\n")
            
            # Write custom atoms header
            f.write("ITEM: ATOMS id type x y z qw qx qy qz\n")
            
            # Write atom attributes
            for n in range(N):
                # Extract from tensor format: qw, qx, qy, qz, x, y, z
                qw, qx, qy, qz, x, y, z = tensor[t, n]
                
                # Format: id (1-indexed), type (always 1), x, y, z, qw, qx, qy, qz
                a_id = n + 1
                f.write(f"{a_id} 1 {x:.5f} {y:.5f} {z:.5f} {qw:.5f} {qx:.5f} {qy:.5f} {qz:.5f}\n")
                
    print("Conversion complete! The file is ready for OVITO.")

# --- Execution ---
if __name__ == "__main__":
    # Ensure the input name matches the test tensor generated previously
    trajectory_npy = "D:\\OneDrive\\OneDrive2\\OneDrive\\OD_Project1\\molecular\\PMMA2\\e_bulk2\\3.1_single_chain\\pmma_trajectory_test.npy"
    save_path = "D:\\OneDrive\\OneDrive2\\OneDrive\\OD_Project1\\molecular\\PMMA2\\e_bulk2\\3.1_single_chain\\pmma_ovito.dump"
    npy_to_lammps_dump(trajectory_npy, save_path)