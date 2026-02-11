# %%
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 1. Define parameters and initial conditions
a_12, a_21 = 3, 1.8 # Parameters
kappa1 = 0.4
kappa2 = 0.4
m = 4
s = 8
r1 = 1
r2 = 1
klim_A = 1
klim_B = 1
V0 = 1
A0 = 1
B0 = 1
T0 = 1
z0 = [0.1, 1]  # Initial conditions: x(0) = 10, y(0) = 5
t_span = (0, 100)  # Time span for the solution

# 2. Define the ODE system as a function
def Vel(t, x, y):
    vel = x**m/(x**m+A0) - y**m/(y**m+B0)
    return vel

def lomakin_ode(t, z):
    """
    z is a list/array where z[0] is x and z[1] is y.
    Returns dz/dt in the same order.
    """
    x, y = z
    V = Vel(t, x, y)
    dxdt = (r1+kappa1*V)*x - klim_A*x**2 - a_12 * x * y
    dydt = (r2-kappa2*V)*y - klim_B*y**2 - a_21 * x * y
    # dxdt = (1+V)*x - x**2 - a_12 * x * y
    # dydt = (1-V)*y - y**2 - a_21 * x * y
    return [dxdt, dydt]

# 3. Solve the ODE
# Pass parameters to the function using the 'args' argument
sol = solve_ivp(
    fun=lomakin_ode,
    t_span=t_span,
    y0=z0,
    dense_output=True
)

# 4. Plot the results
t = np.linspace(t_span[0], t_span[1], 300)
z = sol.sol(t)

plt.figure(figsize=(8, 5))
plt.plot(t, z[0], 'b-', label='A (x)')
plt.plot(t, z[1], 'r--', label='B (y)')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('Lomakin System Solved with solve_ivp')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(t, Vel(t, z[0], z[1]), 'b-', label='V')
plt.xlabel('Time')
plt.ylabel('Normal Velocity')
plt.legend()
plt.grid(True)
plt.show()

print(z[0][-1])