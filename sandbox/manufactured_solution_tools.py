# %%
from sympy import *
from sympy.printing import latex, pretty
from IPython.display import display, Math

def vec_simplify(f):
    "Simplify vector expression."
    return Matrix([simplify(fi) for fi in f])

# Define modified print-related function in case expressions get to large
# to be rendered in MathJax
# Count number of operations

def _count_ops(e):
    if hasattr(e, "shape"):
        return sum(_count_ops(a) for a in e)
    else:
        return e.count_ops()

def eprint(e, name=None, maxsize=500):
    
    """Pretty-print expression, using either LaTeX/MathJax for small
    expressions or SymPy unicode for large expressions, as determined
    by the variable maxsize."""

    if _count_ops(e) <= maxsize:
        s = latex(e)
        if name is not None: s = name + " = " + s
        display(Math(s))

    else:
        s = pretty(e)
        if name is not None: s = name + " = " + s
        disp(s)

# %% [markdown]
# We now define the relevant PDE operators for our problems classes.
# Beside the well-known operators such $\Delta$, $\nabla$ etc. we recall that
# the Laplace-Beltrami operator $\Delta_{\Gamma} u$ can be calculated as follows:
# 
# $$ \Delta_{\Gamma} u = \Delta u - n_{\Gamma} \cdot \nabla^2 u n_{\Gamma} - \textrm{tr}(\nabla n_{\Gamma}) \nabla u \cdot n_{\Gamma},
# $$
# 
# where $n_{\Gamma}$ denotes the surface normal.
# 
# After we haved defined all the necessary operators, we can simply pick our favorite colorful solution
# $u$ and compute corresponding data for the PDE problem at hand. For instance for the **Poisson problem**
# $$
# \begin{align}
#  -\Delta u &= f \quad \text{in } \Omega
#  \\
#  u &= g \quad \text{on } \partial \Omega
# \end{align}
# $$
# we can compute $f$ directly from our chose $u$.
# 
# To manufacture a solution for the **Laplace Beltrami problem**,
# we also need to provide a level-set description $M = \{ \phi = 0 \}$ of our favorite manifold $M$ to compute the normal-field $n_{\Gamma}$ by 
# $$n_{\Gamma} = \dfrac{\nabla\phi}{|\nabla\phi|},$$ 
# as it is part of the definition of $\Delta_{\Gamma}$. Here are the corresponding Python snippets.

# %%
def vec_norm(xx):
    return sum([xi*xi for xi in xx])**0.5

def compute_normal(phi, xx):
    return vec_simplify(Grad(phi, xx)/vec_norm(Grad(phi, xx)))

# Define various differential operators which SymPy does not provide directly.
# TODO: Check whether this is still true!

def Grad(f, x):
    "Compute gradient."
    if not hasattr(f, "shape"):
        return Matrix([diff(f, xj) for xj in x])
    
    elif len(f.shape) == 2 and f.shape[1] == 1:
        return Matrix([[diff(fi, xj) for xj in x] for fi in f])

    else:
        raise RuntimeError("Don't know how to take grad.")

def Project_Grad(f, x, n):
    return Grad(f, x) - (n.dot(Grad(f, x)))*n

def Div(f, x):
    "Compute divergence."

    if len(f.shape) == 2 and f.shape[1] == 1:
        m = f.shape[0]
        return sum([diff(f[i], x[i]) for i in range(m)])

    elif len(f.shape) == 2:
        m, n = f.shape
        return Matrix([sum([diff(f[i, j], x[j]) for j in range(n)])
                       for i in range(m)])

    else:
        raise RuntimeError("Don't know how to take div.")

def Laplace(f, x):
    "Compute the (full) Laplacian"
    
    if not hasattr(f, "shape"):
        return Div(Grad(f,x),x)
    
    else:
        raise RuntimeError("Don't know (yet) how to compute the Laplacian for non-scalar functions.")
    
def Hess(f, x):
    "Compute the Hessian of f"
    n = 1 if not hasattr(x, "shape") else x.shape[0]
    if not hasattr(f, "shape"):
        return Matrix([[diff(diff(f, x[i]), x[j]) for i in range(n)] for j in range(n)])
    else:
        raise RuntimeError("Don't know how to compute the Hessian")

def Tr(f):
    "Compute trace."
    if len(f.shape) == 2 and f.shape[0] == f.shape[1]:
        m = f.shape[0]
        return sum([f[i,i] for i in range(m)])

    else:
        raise RuntimeError("Don't know how to take traces.")

def LaplaceBeltrami(f,x,n):
    "Compute Laplace-Beltrami operator of u"
    return Laplace(f,x) - n.dot(Hess(f,x).dot(n)) - Tr(Grad(n,x))*Grad(f,x).dot(n)

# %% [markdown]
# Next, we define a little wrapper function ```get_solution_str``` which for a given surface (defined by a level set)
# and an analytical reference solution will compute the right-hand side $f$ to the problem
# $$
# -\epsilon\Delta_{\Gamma} u + c u = f \quad \text{on } \Gamma.
# $$
# It will also compute the tangential gradient of the manufactured solution.

# %%
def get_solution_str(u_str, levelset_str, epsilon_str="1", b_str=["0","0","0"], c_str="1"):
    x,y,z = symbols("x y z")
    xx = Matrix([x, y, z])

    phi = eval(levelset_str)
    print("Got the level set function: ", phi)
    normal = compute_normal(phi, xx)
    print('Normal:', normal)
    
    u = parse_expr(u_str)
    print("u: ", u)
    b = Matrix([parse_expr(bb) for bb in b_str])
    print("b: ", b)
    c = parse_expr(c_str)
    print("c: ", c)

    print("Trying to parse epsilon_str")
    epsilon_str="1"
    epsilon = parse_expr(epsilon_str)
    print("epsilon: ", epsilon)
    f = -epsilon*LaplaceBeltrami(u, xx, normal) \
        + c*u \
        + b.dot(Project_Grad(u, xx, normal))
    print('RHS f: ', f)
    
    tang_grad_u = Project_Grad(u, xx, normal)
    return str(f), [str(D) for D in tang_grad_u]

# %% [markdown]
# ## A little example
# 
# Let's compute the $\Delta_{\Gamma} u + u$ for $\Gamma = S^2 = \{\phi = 0\}$ where
# $$
# \begin{cases}
# u(x,y,z) = \sin(\dfrac{\pi x}{2}) \sin(\dfrac{\pi y}{2}) \sin(\dfrac{\pi z}{2})
# \\
# \phi(x,y,z) = x^2 + y^2 + z^2 - 1
# \end{cases}
# $$

# %%
# x,y,z = symbols("x y z")
# xx = Matrix([x, y, z])
# levelset_str = "x**2 + y**2 + z**2 - 1"
# f, tang_grad_u = get_solution_str(u_str, levelset_str)
# 
# eprint(f, name="f")
# eprint(tang_grad_u, name="\\nabla_{\\Gamma}u")

# %% [markdown]
# 


