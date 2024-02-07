import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load CSV file into a Pandas DataFrame
df = pd.read_csv("1.csv")

# Extract relevant columns
X = df['Points:0'].values
Y = df['Points:2'].values
Z = df['Points:3'].values

u_mean = df[['UMean:0', 'UMean:1', 'UMean:2']].values
uprime2mean = df[['UPrime2Mean:0', 'UPrime2Mean:1', 'UPrime2Mean:2', 'UPrime2Mean:3', 'UPrime2Mean:4', 'UPrime2Mean:5']].values
nut = df['nut'].values
p = df['p'].values
wall_shear_stress = df[['wallShearStress:0', 'wallShearStress:1', 'wallShearStress:2']].values
y_plus = df['yPlus'].values


Nx = 51
Ny = 21
Nz = 20

avgProf = np.zeros(Ny)
avgProf_uv = np.zeros(Ny)
avgProf_uw = np.zeros(Ny)
avgProf_nut = np.zeros(Ny)

k = 0
for i in range(0, Nx): 
    for j in range(0, Ny):
        index = k*Nx*Ny + i*Ny + j   # structured grid: i = inner (x); k = outer (z)

        avgProf[j] += u_mean[index,0]/Nx
        avgProf_uv[j] += uprime2mean[index,3]/Nx
        avgProf_uw[j] += uprime2mean[index,4]/Nx
        avgProf_nut[j] += nut[index]/Nx
        

# Perform any additional processing or analysis as needed

# Plot the velocity profile
plt.figure(1)
plt.plot(avgProf[:], Y[0:21], 'r^-')  # Change the indices based on your specific data
plt.xlabel("UMean:0")
plt.ylabel("Points:1 (Y Coordinate)")
plt.title("UMean:0 Velocity Profile")
plt.show()

plt.figure(2)
# plt.plot(u_mean[:, 0], y[:])  # Change the indices based on your specific data
plt.plot(avgProf_uv[:], Y[0:21], 'r^-')  # Change the indices based on your specific data
plt.plot(avgProf_uw[:], Y[0:21], 'k^-')  # Change the indices based on your specific data
plt.xlabel("UMean:0")
plt.ylabel("Points:1 (Y Coordinate)")
plt.title("uv uw")
plt.show()

plt.figure(2)
# plt.plot(u_mean[:, 0], y[:])  # Change the indices based on your specific data
plt.plot(avgProf_nut[:], Y[0:21], 'r^-')  # Change the indices based on your specific data
plt.xlabel("UMean:0")
plt.ylabel("Points:1 (Y Coordinate)")
plt.title("nut")
plt.show()

coordinates = np.zeros(Ny)  # X and Z with all zeros
Y_values = np.array([0.005, 0.00725, 0.0095, 0.01175, 0.014, 0.01625, 0.0185, 0.02075, 0.023, 0.02525,
                     0.0275, 0.02975, 0.032, 0.03425, 0.0365, 0.03875, 0.041, 0.04325, 0.0455, 0.04775, 0.05])

avg_data = {
    'X': np.zeros(Ny),  # All zeros for X
    'Y': Y_values,
    'Z': np.zeros(Ny),  # All zeros for Z
    'avgProfile': avgProf,
    'avgProfile_uv': avgProf_uv,
    'avgProfile_uw': avgProf_uw,
    'avgProfile_nut': avgProf_nut
}

avg_df = pd.DataFrame(avg_data)

# Specify the CSV file name
csv_file = 'averages.csv'

# Write the DataFrame to a CSV file
avg_df.to_csv(csv_file, index=False)
