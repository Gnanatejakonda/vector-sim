import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 1. Set up the page configuration
st.set_page_config(page_title="Basis Transformation Sim", page_icon="📐")

st.title("📐 Vector Basis Transformation")
st.markdown("Enter coordinates below to see the vector in the **Standard Basis (Red)** vs **New Basis (Green)**.")

# 2. Sidebar for User Inputs (Replaces 'input()')
with st.sidebar:
    st.header("1. Input Vector (Standard)")
    vx = st.number_input("Vector X", value=1.0, step=0.1)
    vy = st.number_input("Vector Y", value=0.0, step=0.1)
    
    st.header("2. New Basis Vectors (Green)")
    st.subheader("Basis Vector 1")
    b1x = st.number_input("b1 x-coeff", value=0.0, step=0.1)
    b1y = st.number_input("b1 y-coeff", value=1.0, step=0.1)
    
    st.subheader("Basis Vector 2")
    b2x = st.number_input("b2 x-coeff", value=-1.0, step=0.1)
    b2y = st.number_input("b2 y-coeff", value=0.0, step=0.1)
    
    st.header("3. Origin Shift")
    c1 = st.number_input("Shift X", value=0.0, step=0.5)
    c2 = st.number_input("Shift Y", value=0.0, step=0.5)

# 3. Perform Calculations
v_std = np.array([vx, vy])
basis_matrix = np.array([[b1x, b2x], [b1y, b2y]])

# Check determinant
det = np.linalg.det(basis_matrix)

if abs(det) < 1e-10:
    st.error("⚠️ The basis vectors are Linearly Dependent (Parallel). They cannot span a plane.")
else:
    # Solve system
    v_new_coords = np.linalg.solve(basis_matrix, v_std)
    
    # Calc Angle
    angle_rad = np.arctan2(b1y, b1x)
    angle_deg = np.degrees(angle_rad)
    
    rot_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])

    # 4. Display Text Results
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"**Old Coords (Red):**\n\n {v_std}")
        st.info(f"**Rotation Angle:** {angle_deg:.2f}°")
    with col2:
        st.success(f"**New Coords (Green):**\n\n [{v_new_coords[0]:.2f}, {v_new_coords[1]:.2f}]")
        st.write("**Rotation Matrix:**")
        st.latex(r"\begin{bmatrix} " + f"{rot_matrix[0,0]:.2f} & {rot_matrix[0,1]:.2f} \\\\ {rot_matrix[1,0]:.2f} & {rot_matrix[1,1]:.2f}" + r" \end{bmatrix}")

    # 5. Visualization (Matplotlib)
    fig, ax = plt.subplots(figsize=(8, 8))
    
    limit = max(np.linalg.norm(v_std), 2.0) * 1.5
    ax.set_xlim(-limit + c1, limit + c1)
    ax.set_ylim(-limit + c2, limit + c2)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    origin = np.array([c1, c2])

    def draw_vec(vec, origin, color, label):
        ax.quiver(origin[0], origin[1], vec[0], vec[1], 
                  angles='xy', scale_units='xy', scale=1, 
                  color=color, label=label)

    # Draw Vectors
    draw_vec([1, 0], origin, 'red', 'Old X')
    draw_vec([0, 1], origin, 'red', 'Old Y')
    draw_vec(v_std, origin, 'red', 'Vector Old')
    
    draw_vec([b1x, b1y], origin, 'green', 'New b1')
    draw_vec([b2x, b2y], origin, 'green', 'New b2')

    # Annotations
    ax.text(origin[0] + vx, origin[1] + vy, f" Old", color='red', fontweight='bold')
    ax.text(origin[0] + vx, origin[1] + vy - (limit*0.05), f" New", color='green', fontweight='bold')

    plt.legend(loc='lower right')
    plt.title("Vector Simulation")
    
    # 6. Push the plot to the web page
    st.pyplot(fig)