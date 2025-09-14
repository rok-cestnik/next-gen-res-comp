import matplotlib.pyplot as plt
from math import log

system = 'lorenz'

plt.rcParams.update({"text.usetex": True, "font.size": 14, "font.family": "serif"})

## NOISE 0
# Read data from file
N, ert, erv = [], [], []
with open("saves/"+system+"_errors_0.txt", "r") as file:
    for line in file:
        values = line.strip().split(",")  # Assuming CSV format
        if len(values) == 3:
            N.append(float(values[0]))
            ert.append(float(values[1]))
            erv.append(float(values[2]))

N0len = len(N)

# Plot the data
plt.scatter(N, ert, s=5, label="Training")
plt.scatter(N, erv, s=2, label="Validation")
plt.xlabel(r"$M$")
plt.ylabel(r"$E$")
plt.yscale("log")  # Set y-axis to log scale
#plt.xscale("log")  # Set y-axis to log scale
plt.legend()
#plt.title("Plot of x vs y1 and x vs y2")
plt.savefig("pics/"+system+"_plot_0.png",dpi=300, bbox_inches='tight')
plt.show()


## NOISE 001
# Read data from file
N, ert, erv = [], [], []
with open("saves/"+system+"_errors_001.txt", "r") as file:
    for line in file:
        values = line.strip().split(",")  # Assuming CSV format
        if len(values) == 3:
            N.append(float(values[0]))
            ert.append(float(values[1]))
            erv.append(float(values[2]))

# Plot the data
plt.scatter(N, ert, s=5, label="Training")
plt.scatter(N, erv, s=2, label="Validation")
plt.xlabel(r"$M$")
plt.ylabel(r"$E$")
plt.yscale("log")  # Set y-axis to log scale
#plt.xscale("log")  # Set y-axis to log scale
plt.legend()
#plt.title("Plot of x vs y1 and x vs y2")
plt.savefig("pics/"+system+"_plot_001.png",dpi=300, bbox_inches='tight')
plt.show()


## NOISE 001 again
# Read data from file
N, ert, erv = [], [], []
with open("saves/"+system+"_errors_001_again.txt", "r") as file:
    for line in file:
        values = line.strip().split(",")  # Assuming CSV format
        if len(values) == 3:
            N.append(float(values[0]))
            ert.append(float(values[1]))
            erv.append(float(values[2]))

# Plot the data
plt.scatter(N[0:N0len], ert[0:N0len], s=5, label="Training")
plt.scatter(N[0:N0len], erv[0:N0len], s=2, label="Validation")
plt.xlabel(r"$M$")
plt.ylabel(r"$E$")
plt.yscale("log")  # Set y-axis to log scale
#plt.xscale("log")  # Set y-axis to log scale
plt.legend()
#plt.title("Plot of x vs y1 and x vs y2")
plt.savefig("pics/"+system+"_plot_001_again.png",dpi=300, bbox_inches='tight')
plt.show()


## NOISE 0003
# Read data from file
N, ert, erv = [], [], []
with open("saves/"+system+"_errors_0003.txt", "r") as file:
    for line in file:
        values = line.strip().split(",")  # Assuming CSV format
        if len(values) == 3:
            N.append(float(values[0]))
            ert.append(float(values[1]))
            erv.append(float(values[2]))

# Plot the data
plt.scatter(N, ert, s=5, label="Training")
plt.scatter(N, erv, s=2, label="Validation")
plt.xlabel(r"$M$")
plt.ylabel(r"$E$")
plt.yscale("log")  # Set y-axis to log scale
#plt.xscale("log")  # Set y-axis to log scale
plt.legend()
#plt.title("Plot of x vs y1 and x vs y2")
plt.savefig("pics/"+system+"_plot_0003.png",dpi=300, bbox_inches='tight')
plt.show()


## NOISE 002
# Read data from file
N, ert, erv = [], [], []
with open("saves/"+system+"_errors_002.txt", "r") as file:
    for line in file:
        values = line.strip().split(",")  # Assuming CSV format
        if len(values) == 3:
            N.append(float(values[0]))
            ert.append(float(values[1]))
            erv.append(float(values[2]))

# Plot the data
plt.scatter(N, ert, s=5, label="Training")
plt.scatter(N, erv, s=2, label="Validation")
plt.xlabel(r"$M$")
plt.ylabel(r"$E$")
plt.yscale("log")  # Set y-axis to log scale
#plt.xscale("log")  # Set y-axis to log scale
plt.legend()
#plt.title("Plot of x vs y1 and x vs y2")
plt.savefig("pics/"+system+"_plot_002.png",dpi=300, bbox_inches='tight')
plt.show()
