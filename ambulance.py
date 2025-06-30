import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# Simulation parameters
signal_location = 0  # Traffic signal at 0 km
ambulance_position = 2.0  # Initial ambulance position (2 km away)

# Create a figure and axis
fig, ax = plt.subplots()
plt.title("Ambulance & Traffic Signal Simulation")
ax.set_xlim(-0.5, 2.5)
ax.set_ylim(0, 1)
ax.set_yticks([])

# Signal light and ambulance marker
signal_light = plt.Circle((signal_location, 0.5), 0.1, color='red')
ambulance_marker, = ax.plot([], [], marker="s", markersize=15, color='blue', label="Ambulance")

# Add signal light to plot
ax.add_patch(signal_light)
ax.legend(loc='upper right')

def update(frame):
    global ambulance_position

    # Move ambulance closer
    ambulance_position -= random.uniform(0.05, 0.2)
    if ambulance_position < 0:
        ambulance_position = 2.0  # Reset position after passing the signal

    # Update ambulance position (wrap in lists!)
    ambulance_marker.set_data([ambulance_position], [0.5])

    # Check distance to signal
    distance = abs(ambulance_position - signal_location)
    if distance <= 1.0:
        signal_light.set_color('green')  # Give way to ambulance
    else:
        signal_light.set_color('red')  # Default state

    return ambulance_marker, signal_light

# Create animation
ani = animation.FuncAnimation(fig, update, frames=100, interval=500, blit=True)

plt.show()
