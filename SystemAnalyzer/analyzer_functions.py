import sys
sys.path.append('..')
from include import *

# Functions used in the SystemAnalyzer. 

def get_path(main_path="/home/danie/erda", sub_path=""):
    """Returns the directory of a subfolder"""
    return os.path.join(main_path, sub_path)

def load_npy(fname, verbose=False):
    if verbose:
        print(f"Loading Data from: {fname}")
    data = np.load(fname)
    return data

def get_max_frame(calm_dir):
    """Finds the last frame in a given CALM directory"""
    frame_names = get_CALM_frame_names(calm_dir, filter="mean") # We can use any filter here, since we just want to know the number of frames, and all files should have the same number of frames
    return len(frame_names)


def load_dat(fname, verbose=False):
    if verbose:
        print(f"Loading Data from: {fname}")
    data = np.loadtxt(fname)
    return data

def load_mda_universe(fname_gro, fname_xtc):
    """Function for simplifying the load of an mda universe. 
    Provide a path to a topology file (.gro works) and trajectory file (.xtc works)"""
    system = mda.Universe(fname_gro, fname_xtc, in_memory=False)
    return system

def load_CALM_dimensions(dim_fname):
    """Small function for loading the dimensions.csv file.
    This both contains the changing boxsizes but also which frames have been analyzed.
    RETURNS:
    trj_idxs: The indexes for the trajectory
    boxsizes: a list of the boxsizes for each frame"""
    data_raw = pd.read_csv(dim_fname, delimiter=",", skiprows=1, header=None)         # Box size now contains the boundary for each frame 
                                                                                     # AND also contains the frame indexes which is very nice
    data_raw.columns = ["trajectory_idx", "x", "y", "z"] # Rename the columns for easier access
    trj_idxs = data_raw["trajectory_idx"].values - 1 # Mapping from 1-based to 0-based indexing
    boxsizes = data_raw[["x", "y", "z"]].values
    return trj_idxs, boxsizes

def get_CALM_frame_names(calm_dir, filter):
    """Gets the names of all frames for which CALM curvature maps are available, by looking at the files in the CALM directory.
    This is useful to make sure we only analyze frames for which we have curvature data.
    ARGS:
        calm_dir: the directory where the CALM curvature maps are stored
        filter: a string to filter the files by, e.g. "mean" to only select mean curvatues
    """
    files = os.listdir(calm_dir)
    frame_names = set()
    for file in files:
        if file.endswith(".npy") and filter in file:
            frame_names.add(os.path.join(calm_dir, file))
    return sorted(list(frame_names))


def pos_to_curvature(curvature, positions, boxsize):
    # Mapping the location to a specific grid in the curvature array (curv.shape(100,100))
    idxs = positions / boxsize * 100    # Mapping it from the boxsize to a number between 0 and 100
    idxs = np.array(idxs, dtype=int)    # Mapping the float number to lowest int
    idxs = idxs[:, :-1] # Only keeping the xy dimensions
    idxs = np.clip(idxs, 0, 99) # Mapping the values directly to integers bewteen 0 and 99
    curv_lipid = curvature[idxs[:, 1], idxs[:, 0]]
    return curv_lipid


def load_xvg_file(fname):
    """A simple function for loading .xvg files. Loads the data from the analysis_files folder by default, 
    so the only thing we need to change is the specific run from which we then have acccess to all the different .xvg files. 
    Returns a 2D array with the x and y data."""
    data = np.loadtxt(fname, comments=['@', '#'])
    x, y = data.T


    return np.array([x, y])


def plot_raw_file(fname, path, title, xlabel, ylabel, color='blue', xlim=None, ylim=None, figsize=(8, 6)):
    """A simple function for plotting raw .xvg files with basic formatting."""
    data = load_xvg_file(fname, path)
    x, y = data

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    ax.grid()
    



def load_leaflets_idx(fname):
    """Loads the leaflet.ndx file and returns a dictionary with the leaflet names as keys and the indices as values"""
    print(f"Loading Leaflets from: {fname}")
    with open(fname, 'r') as f:
        lines = f.readlines()
    
    leaflet_dict = {}
    current_leaflet = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('['):
            current_leaflet = line[1:7].strip()  # Extract leaflet name
            leaflet_dict[current_leaflet] = []
        elif current_leaflet is not None and line:
            indices = list(map(int, line.split()))
            indices = np.array(indices) - 1  # Mapping from 1-based to 0-based indexing)
            leaflet_dict[current_leaflet].extend(indices)
    
    return leaflet_dict

def calc_surface_area(boxsize, membrane_height):
    """Calculates the surface area of each grid from the parametrizable surface element dA = np.length(np.cross(r_x, r_y))*dx*dy. 
    Scales the length according to the boxsize"""
    Nx, Ny = membrane_height.shape
    # What is the difference between each grid point
    dx = boxsize[0]/Nx # The height is in nm, but the boxsize is in Angstrom, so we need to convert it to nm
    dy = boxsize[1]/Ny
    
    # Calculate the gradient from a central scheme for the in between points, and a forward/backward scheme at edges
    hy, hx = np.gradient(membrane_height, dy, dx)    
    
    # Compute the area of each element according to the area of a parametrizable surface
    grid_area = np.sqrt(1+hx**2+hy**2)*(dx*dy)
    return grid_area, hx, hy


def create_pdf_err(data, n_bins, bin_range):
    """Creates the bin_centers, density and the poisson errors for a density=True histogram. This allows to create errorbars on a pdf"""
    # Poisson errors (These occur since all measurements are independent
    # so the expected value in a specific bin is given by its small probability of being in that exact bin)
    density, bins = np.histogram(data, bins=n_bins, density=True, range=bin_range)
    bin_centers = (bins[1:] + bins[:-1]) / 2
    bw = bins[1] - bins[0]
    pdf_err = np.sqrt(density/(len(data) * bw))      # Since the distribution is a pdf, 
                                                     # the poisson errors also needs to be rescaled 
                                                     # slightly: \sigma_p = \sqrt{p/(N \Delta x)}
                                                                        
    return bin_centers, density, pdf_err # This allows to create errorbars on a pdf