import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

st.set_page_config(
	page_title = 'A buckling beam',
	page_icon = "📝",
	layout = "centered"
	)

st.title('Buckling of a beam with fixed supports')

st.header("Problem description")
col1, col2 = st.columns([3,2])
col1.write("""&nbsp;&nbsp;&nbsp;&nbsp; An inextensible beam, *elastica*, of length $L$ is clamped at the two ends, and it is compressed with controlled displacement $\Delta$ at one end, shown in figure on the right.
	""")
col2.image("pb_description.png", caption='An elastica deforms under controlled displacement.')



def sys_equ(s, var, nx, ny): 
    th, dth, x, y = var
    eq0 = dth
    eq1 = nx*np.sin(th) - ny*np.cos(th)
    eq2 = np.cos(th)
    eq3 = np.sin(th)
    return [eq0, eq1, eq2, eq3]


@st.cache_data
def load_data(url):
	bvp_mode1 = pd.read_csv("%s/solve_bvp_data_mode1.csv"%url)
	bvp_mode2 = pd.read_csv("%s/solve_bvp_data_mode2.csv"%url)
	mode12_bvp = pd.read_csv("%s/Data_mode12_bvp.csv"%url)
	return bvp_mode1, bvp_mode2, mode12_bvp


bvp_mode1, bvp_mode2, mode12_bvp = load_data("./Data")

z1 = np.concatenate((np.flip(bvp_mode1.dtheta0.values), -bvp_mode1.dtheta0.values), axis=None)
x1 = np.concatenate((np.flip(bvp_mode1.Nx.values), bvp_mode1.Nx.values), axis=None)
y1 = np.concatenate((np.flip(bvp_mode1.Ny.values), -bvp_mode1.Ny.values), axis=None)

z2 = np.concatenate((np.flip(bvp_mode2.dtheta0.values), -bvp_mode2.dtheta0.values), axis=None)
x2 = np.concatenate((np.flip(bvp_mode2.Nx.values), bvp_mode2.Nx.values), axis=None)
y2 = np.concatenate((np.flip(bvp_mode2.Ny.values), -bvp_mode2.Ny.values), axis=None)

z12 = np.concatenate((mode12_bvp.dtheta0.values,np.flip(- mode12_bvp.dtheta0.values)), axis=None)
x12 = np.concatenate((mode12_bvp.Nx.values,np.flip(mode12_bvp.Nx.values)), axis=None)
y12 = np.concatenate((mode12_bvp.Ny.values,np.flip(mode12_bvp.Ny.values)), axis=None)
z12 = np.concatenate((z12,np.flip(z12)), axis=None)
x12 = np.concatenate((x12, np.flip(x12)), axis=None)
y12 = np.concatenate((y12,np.flip(-y12)), axis=None)

fig = plt.figure(figsize=(8, 18))
ax = fig.add_subplot(121, projection='3d')
ax.plot(x1, y1, z1, label='Mode1')
ax.plot(x2, y2, z2, label='Mode2')
ax.plot(x12, y12, z12, label='Mode1/2')
# ax.set_title('')
ax.set_xlabel("Nx")
ax.set_ylabel("Ny")
ax.set_zlabel(r"$d\theta/ds(0)$", labelpad=.5)
ax.legend()


ax2 = fig.add_subplot(122)
ax2.set_xlabel("x")
ax2.set_ylabel("y", labelpad=.1)
ax2.grid(True)
ax2.set_aspect('equal') 
ax2.set_xlim([-0.5, 1])
ax2.set_ylim([-0.2, 0.5])

plt.tight_layout(w_pad = 4.8)

# Mesh points for s
s = np.linspace(0, 1, 100)

col1.markdown(r"""
&nbsp;&nbsp;&nbsp;&nbsp; The system of differential equations is:
$$
\left\{
\begin{aligned}
	& EI\theta''(s) = N_x \sin{\theta(s)} - N_y \cos{\theta(s)}\\
	& x'(s) = \cos{\theta}(s)\\
	& y'(s) = \sin{\theta}(s)
\end{aligned}
\right.
$$
where $N_x$ and $N_y$ are two parameters to be found with $\theta(s)$, $x(s)$ and $y(s)$. 

""")

r"""
The boundary conditions are:

$$x(0) = y(0) = y(L) =0 \qquad   x(L) = L - \Delta \qquad \theta(0) = \theta(L) = 0$$
"""

with st.expander("Dimentionless system"):
    st.markdown(r"""When doing numerical calculation, the system is adimentionlize by length scale $L$ and force scale $EI/L^2$.""")



"""
***
"""
st.header("Solving methods")
st.markdown(r"""

With the help of Python library `scipy`, 2 methods are implemented to solve the differential equations above:

1. bvp solver: using `scipy.integrate.solve_bvp` to solve the system of differential equations as a boundary value problem.
2. shoting method + ivp solver: using `solve_ivp` and `scipy.optimize.root` to find appropriate values for $\left.\frac{d\, \theta(s)}{d\,s}\right|_{s=0}$, $N_x$ and $N_y$ such that solving the equations as an initial value problem gives a solution which satisfies the boundary condition at the other end ($s=L$).


### Remarks 
Both `solve_bvp` and `minimize` need values over the whole beam to define initial state, whereas `root` +`solve_ivp` only needs 3 values: $d\theta(0)/ds$, $N_x$ and $N_y$.
""")






st.header("Numerical solving results")

with st.sidebar:
	st.write("Select a specific mode : ")
	#create a selectbox option that choose mode
	selected_mode = st.selectbox(' ', ['Mode1', 'Mode2', 'Mode1/2'], 0)

if selected_mode == 'Mode1':
	new_delta_value = st.slider(label = "Delta: ",
		min_value = 0.0001, max_value = 1.45, value = 0.1)
	point_mode = bvp_mode1.iloc[(bvp_mode1['delta'] - new_delta_value).abs().argsort()[0]]
	ax2.set_title("Form of the beam - mode1")

elif selected_mode == 'Mode2':
	new_delta_value = st.slider(label = "Delta: ",
		min_value = 0.0001, max_value = 1.45, value = 0.1)
	point_mode = bvp_mode2.iloc[(bvp_mode2['delta'] - new_delta_value).abs().argsort()[0]]
	ax2.set_title("Form of the beam - mode2")

else : 
	new_delta_value = st.slider(label = "Nx: ",
		min_value = -86, max_value = 56, value = -44)
	point_mode = mode12_bvp.iloc[(mode12_bvp['Nx'] - new_delta_value).abs().argsort()[0]]
	ax2.set_title("Form of the beam - mode1/2")


solution_ivp = solve_ivp(sys_equ, [0, 1], [0, point_mode['dtheta0'],0, 0], args=(point_mode['Nx'], point_mode['Ny']), dense_output=True)

line, = ax2.plot(solution_ivp.sol(s)[2], solution_ivp.sol(s)[3])
sc = ax.scatter([point_mode['Nx']], [point_mode['Ny']], [point_mode['dtheta0']], color = 'red',s=10, label='_nolegend_')


st.pyplot(fig)




col1, col2, col3 = st.columns(3)
col1.metric("$N_x$", "%.4f"%point_mode['Nx'])
col2.metric("$N_y$", "%.4f"%point_mode['Ny'])
col3.metric(r"$\frac{d \theta}{ds}(0)$", "%.4f"%point_mode['dtheta0'])





















