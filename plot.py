import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from decimal import Decimal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider # For standalone script interactivity

def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]

with open("times.pkl", "rb") as f:
    times = pickle.load(f)
    
x = []
y = []
z_h = []
z_o = []

for p in times:
    x.append(p["num_cars"])
    y.append(p["num_chargers"])
    z_h.append(p["h"])
    z_o.append(p["o"])
    # z.append(p["optimal_n_max"] * p["optimal_ls_max"])

x = np.array(x)
y = np.array(y)
z_h = np.array(z_h)
z_o = np.array(z_o)

def preprocess_data_for_slicing(x_data, y_data, z_data):
    """
    Organizes data into slices based on unique x-values.

    Args:
        x_data (np.array): 1D array of x-coordinates.
        y_data (np.array): 1D array of y-coordinates.
        z_data (np.array): 1D array of z-coordinates.

    Returns:
        tuple: (sorted_unique_x, data_by_x_slice)
               sorted_unique_x: A sorted numpy array of unique x-values.
               data_by_x_slice: A dictionary where keys are unique x-values
                                and values are tuples (y_slice, z_slice).
    """
    unique_x_values = np.unique(x_data) # Automatically sorts them
    
    data_by_x_slice = {}
    for x_val in unique_x_values:
        # Find indices where x_data matches the current unique x_val
        indices = np.where(x_data == x_val)[0]
        data_by_x_slice[x_val] = (y_data[indices], z_data[indices])
        
    return unique_x_values, data_by_x_slice

def plot_yz_slice(ax, slice_x_value, y_coords_slice, z_coords_slice, slice_index):
    """
    Plots a single y-z slice on a given matplotlib axes.
    """
    ax.clear() # Clear previous plot on this axis
    if len(y_coords_slice) > 0 and len(z_coords_slice) > 0:
        ax.scatter(y_coords_slice, z_coords_slice, s=10, alpha=0.7) # s is marker size
        # You could also use:
        # ax.plot(y_coords_slice, z_coords_slice, 'o', markersize=3)
    else:
        ax.text(0.5, 0.5, "No data for this slice", ha='center', va='center', transform=ax.transAxes)
        
    ax.set_xlabel("Y coordinate")
    ax.set_ylabel("Z coordinate")
    ax.set_title(f"Y-Z plane at X = {slice_x_value:.2f} (Index: {slice_index})")
    ax.grid(True)
    ax.axis('equal')

fig, ax = plt.subplots(figsize=(8, 7))
plt.subplots_adjust(left=0.1, bottom=0.25) # Make room for the slider

x, y = y, x

sorted_unique_x, preprocessed_slices = preprocess_data_for_slicing(x, y, z_o)

# Initial plot (first slice)
initial_slice_index = 0
initial_x_value = sorted_unique_x[initial_slice_index]
y_initial_slice, z_initial_slice = preprocessed_slices[initial_x_value]
plot_yz_slice(ax, initial_x_value, y_initial_slice, z_initial_slice, initial_slice_index)

# Create the slider axis
ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03], facecolor='lightgoldenrodyellow')

# Create the slider
# valmin is 0, valmax is number of slices - 1, valinit is initial_slice_index
# valstep is 1 because we want integer indices
slider_x_index = Slider(
    ax=ax_slider,
    label='X-Slice Index',
    valmin=0,
    valmax=len(sorted_unique_x) - 1,
    valinit=initial_slice_index,
    valstep=1
)

# Update function to be called when slider value changes
def update(val):
    slice_index = int(slider_x_index.val) # Get integer index from slider
    current_x_value = sorted_unique_x[slice_index]
    y_current_slice, z_current_slice = preprocessed_slices[current_x_value]
    plot_yz_slice(ax, current_x_value, y_current_slice, z_current_slice, slice_index)
    fig.canvas.draw_idle() # Redraw the figure

# Attach the update function to the slider
slider_x_index.on_changed(update)

plt.show()
