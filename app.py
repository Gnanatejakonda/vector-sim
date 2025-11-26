import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 1. Set up the page configuration
st.set_page_config(page_title="Basis Transformation Sim", page_icon="📐")

st.title("📐 Vector Basis Transformation")
st.markdown("Enter coordinates below to see the vector in the **Standard Basis (Blue)** vs **New Basis (Green)**.")

# 2. Sidebar for User Inputs
with st.sidebar:
    st.header("1. Input Vector (Standard)")
    vx = st.number_input("Vector X", value=2.0, step=0.1)
    vy = st.number_input("Vector Y", value=1.0, step=0.1)
    
    st.header("2. New Basis Vectors (Green)")
    st.subheader("Basis Vector 1")
    b1x = st.number_input("b1 x-coeff", value=1.0, step=0.1)
    b1y = st.number_input("b1 y-coeff", value=1.0, step=0.1)
    
    st.subheader("Basis Vector 2")
    b2x = st.number_input("b2 x-coeff", value=-0.5, step=0.1)
    b2y = st.number_input("b2 y-coeff", value=1.0, step=0.1)
    
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
    
    # Calc Angle for display
    angle_rad = np.arctan2(b1y, b1x)
    angle_deg = np.degrees(angle_rad)
    
    rot_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])

    # 4. Display Text Results
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Old Coords (Blue):**\n\n {v_std}")
        st.write(f"**Rotation Angle:** {angle_deg:.2f}°")
    with col2:
        st.success(f"**New Coords (Green):**\n\n [{v_new_coords[0]:.2f}, {v_new_coords[1]:.2f}]")
        st.write("**Rotation Matrix:**")
        st.latex(r"\begin{bmatrix} " + f"{rot_matrix[0,0]:.2f} & {rot_matrix[0,1]:.2f} \\\\ {rot_matrix[1,0]:.2f} & {rot_matrix[1,1]:.2f}" + r" \end{bmatrix}")

    # 5. Visualization (Matplotlib)
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Calculate limits to ensure everything fits
    limit = max(np.linalg.norm(v_std), 3.0) * 1.5
    ax.set_xlim(-limit + c1, limit + c1)
    ax.set_ylim(-limit + c2, limit + c2)
    ax.set_aspect('equal')
    
    # Hide default grid, we will draw our own axes
    ax.grid(False)
    
    origin = np.array([c1, c2])

    # --- DRAWING EXTENDED AXES (Background Lines) ---
    
    # 1. Old Axes (Standard X/Y) - Grey Dashed
    ax.axhline(y=c2, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=c1, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # 2. New Axes (Green lines extending to edge)
    # We plot a long line passing through origin with slope of b1 and b2
    scale_factor = 100 # Make lines very long
    
    # Axis 1 Line
    ax.plot([c1 - scale_factor*b1x, c1 + scale_factor*b1x], 
            [c2 - scale_factor*b1y, c2 + scale_factor*b1y], 
            color='green', alpha=0.3, linestyle='-', linewidth=1)
            
    # Axis 2 Line
    ax.plot([c1 - scale_factor*b2x, c1 + scale_factor*b2x], 
            [c2 - scale_factor*b2y, c2 + scale_factor*b2y], 
            color='green', alpha=0.3, linestyle='-', linewidth=1)


    # --- DRAWING VECTORS (Thick Arrows) ---
    
    def draw_vec(vec, origin, color, label, thickness=0.005):
        ax.quiver(origin[0], origin[1], vec[0], vec[1], 
                  angles='xy', scale_units='xy', scale=1, 
                  color=color, label=label, width=thickness)

    # Draw Basis Vectors (Medium Thickness)
    draw_vec([b1x, b1y], origin, 'green', 'New Basis b1', thickness=0.012)
    draw_vec([b2x, b2y], origin, 'green', 'New Basis b2', thickness=0.012)
    
    # Draw Main Vector (Very Thick & Blue)
    draw_vec(v_std, origin, 'blue', 'Vector (Blue)', thickness=0.02)

    # Annotations
    ax.text(origin[0] + vx, origin[1] + vy, f"  ({vx}, {vy})", color='blue', fontweight='bold', fontsize=12)
    
    # Add a small legend
    plt.legend(loc='lower right')
    plt.title("Visual Simulation")
    
    # 6. Push the plot to the web page
    st.pyplot(fig)
