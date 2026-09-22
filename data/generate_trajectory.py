import numpy as np

def vec_to_quat(d):
    """
    Converts a normalized direction vector to a quaternion (qw, qx, qy, qz).
    Assumes the reference vector is (0, 0, 1).
    """
    d = d / np.linalg.norm(d)
    dx, dy, dz = d
    
    if dz < -0.999999:
        return np.array([0.0, 1.0, 0.0, 0.0]) 
    
    qw = 1.0 + dz
    qx = -dy
    qy = dx
    qz = 0.0
    
    norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    return np.array([qw, qx, qy, qz]) / norm

# Added 'max_frames' parameter with a default of None
def extract_pmma_tensor(data_file, dump_file, output_npy, max_frames=None):
    # 1. Parse topology from the data file
    backbone_atoms = []
    side_atoms = []
    
    print("Parsing topology...")
    with open(data_file, 'r') as f:
        bond_section = False
        for line in f:
            line = line.strip()
            if line.startswith("Bonds"):
                bond_section = True
                continue
            if bond_section:
                if line.startswith("Angles") or (line and not line[0].isdigit() and "source" not in line):
                    if not line.startswith("[source"): 
                        break
                
                parts = line.split()
                if len(parts) >= 4 and parts[0].isdigit():
                    b_type = int(parts[1])
                    a1 = int(parts[2])
                    a2 = int(parts[3])
                    
                    if b_type == 2:  # Side chain bond
                        b_atom = min(a1, a2) # Odd ID (Backbone)
                        s_atom = max(a1, a2) # Even ID (Side chain)
                        
                        backbone_atoms.append(b_atom)
                        side_atoms.append(s_atom)

    sorted_indices = np.argsort(backbone_atoms)
    backbone_atoms = np.array(backbone_atoms)[sorted_indices]
    side_atoms = np.array(side_atoms)[sorted_indices]
    
    print(f"Identified {len(backbone_atoms)} backbone atoms.")

    # 2. Parse the dump file frame-by-frame
    trajectory = []
    frames_processed = 0  # Initialize a counter
    
    print("Extracting frames from dump file...")
    
    with open(dump_file, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break # EOF reached
            
            if "ITEM: TIMESTEP" in line:
                timestep = int(f.readline().strip())
                
                f.readline() # ITEM: NUMBER OF ATOMS
                num_atoms = int(f.readline().strip())
                
                f.readline() # ITEM: BOX BOUNDS
                f.readline()
                f.readline()
                f.readline()
                
                f.readline() # ITEM: ATOMS id type x y z ix iy iz
                
                coords = {}
                for _ in range(num_atoms):
                    parts = f.readline().split()
                    a_id = int(parts[0])
                    coords[a_id] = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
                
                frame_data = []
                for b_id, s_id in zip(backbone_atoms, side_atoms):
                    pos = coords[b_id]
                    side_pos = coords[s_id]
                    
                    vec = side_pos - pos 
                    q = vec_to_quat(vec)
                    
                    frame_data.append([q[0], q[1], q[2], q[3], pos[0], pos[1], pos[2]])
                
                trajectory.append(frame_data)
                frames_processed += 1  # Increment the counter
                
                # Log progress more frequently for testing
                if frames_processed % 100 == 0:
                    print(f"Processed {frames_processed} frames...")
                
                # Break the loop if we hit the requested limit
                if max_frames is not None and frames_processed >= max_frames:
                    print(f"Reached requested limit of {max_frames} frames. Stopping extraction.")
                    break

    # 3. Compile the final tensor and save
    tensor = np.array(trajectory)
    print(f"Extraction complete! Tensor shape: {tensor.shape} (T, N, 7)")
    
    np.save(output_npy, tensor)
    print(f"Tensor saved to {output_npy}")

# --- Execution ---
if __name__ == "__main__":
    # Passing 100 to the max_frames argument
    data_file = "D:\\OneDrive\\OneDrive2\\OneDrive\\OD_Project1\\molecular\\PMMA2\\e_bulk2\\3.1_single_chain\\PMMA_data_initial.data"
    dump_file = "D:\\OneDrive\\OneDrive2\\OneDrive\\OD_Project1\\molecular\\PMMA2\\e_bulk2\\3.1_single_chain\\out.txt"
    save_path = "D:\\OneDrive\\OneDrive2\\OneDrive\\OD_Project1\\molecular\\PMMA2\\e_bulk2\\3.1_single_chain\\pmma_trajectory_full.npy"
    extract_pmma_tensor(data_file, dump_file, save_path)#, max_frames=10000)