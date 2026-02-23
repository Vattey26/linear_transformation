import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches

def create_square():
    """Create a unit square"""
    return np.array([[0, 1, 1, 0, 0],
                     [0, 0, 1, 1, 0]])

def create_house():
    """Create a house shape"""
    house = np.array([[0, 1, 1, 2, 3, 4, 4, 3, 2, 0, 0],
                      [0, 0, 2, 2, 3, 2, 2, 0, 0, 0, 0]])
    return house * 0.3

def rotation_matrix(angle_degrees):
    """Create rotation matrix"""
    theta = np.radians(angle_degrees)
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])

def scaling_matrix(sx, sy):
    """Create scaling matrix"""
    return np.array([[sx, 0],
                     [0, sy]])

def shearing_matrix(k, horizontal=True):
    """Create shearing matrix"""
    if horizontal:
        return np.array([[1, k],
                        [0, 1]])
    else:
        return np.array([[1, 0],
                        [k, 1]])

def reflection_matrix(axis='x'):
    """Create reflection matrix"""
    if axis == 'x':
        return np.array([[1, 0],
                        [0, -1]])
    elif axis == 'y':
        return np.array([[-1, 0],
                        [0, 1]])
    elif axis == 'xy':
        return np.array([[0, 1],
                        [1, 0]])

def plot_transformation(original_shape, transformed_shape, T, title_text):
    """Plot original and transformed shapes with analysis"""
    
    fig = plt.figure(figsize=(14, 6))
    
    # Calculate properties
    det_T = np.linalg.det(T)
    eigenvalues, eigenvectors = np.linalg.eig(T)
    
    # Plot 1: Original shape
    ax1 = plt.subplot(1, 2, 1)
    ax1.fill(original_shape[0, :], original_shape[1, :], 'b', alpha=0.3, 
             edgecolor='b', linewidth=2, label='Shape')
    ax1.plot([0, 1], [0, 0], 'r-', linewidth=2, label='x-axis')
    ax1.plot([0, 0], [0, 1], 'g-', linewidth=2, label='y-axis')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    ax1.set_aspect('equal')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Original Shape', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    
    # Plot 2: Transformed shape
    ax2 = plt.subplot(1, 2, 2)
    ax2.fill(transformed_shape[0, :], transformed_shape[1, :], 'r', alpha=0.3,
             edgecolor='r', linewidth=2, label='Transformed')
    
    # Show transformed basis vectors
    transformed_x = T @ np.array([1, 0])
    transformed_y = T @ np.array([0, 1])
    ax2.arrow(0, 0, transformed_x[0], transformed_x[1], 
             head_width=0.15, head_length=0.15, fc='r', ec='r', 
             linewidth=2, label='Transformed x-axis')
    ax2.arrow(0, 0, transformed_y[0], transformed_y[1],
             head_width=0.15, head_length=0.15, fc='g', ec='g',
             linewidth=2, label='Transformed y-axis')
    
    # Plot eigenvectors if they are real
    if np.all(np.isreal(eigenvalues)):
        for i in range(2):
            ev = eigenvectors[:, i].real
            lambda_val = eigenvalues[i].real
            
            if np.abs(lambda_val) > 0.01:  # Only plot if eigenvalue is significant
                # Plot eigenvector direction
                ax2.arrow(0, 0, ev[0], ev[1],
                         head_width=0.15, head_length=0.15,
                         fc='magenta', ec='magenta', linewidth=2.5,
                         linestyle='--', alpha=0.7)
                
                # Plot scaled eigenvector
                scaled_ev = lambda_val * ev
                ax2.arrow(0, 0, scaled_ev[0], scaled_ev[1],
                         head_width=0.15, head_length=0.15,
                         fc='cyan', ec='cyan', linewidth=2, alpha=0.7)
    
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)
    ax2.set_aspect('equal')
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.set_title(title_text, fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    
    # Main title
    fig.suptitle(f'2D Linear Transformation Visualizer\nDeterminant = {det_T:.3f} (Area scaling factor)',
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Print information
    print("\n" + "="*50)
    print("TRANSFORMATION MATRIX:")
    print("="*50)
    print(T)
    
    print("\n" + "="*50)
    print("LINEAR ALGEBRA PROPERTIES:")
    print("="*50)
    print(f"Determinant: {det_T:.4f}")
    print(f"  (Area scaling factor: original area × {abs(det_T):.4f} = transformed area)")
    
    print("\nEigenvalues:")
    print(eigenvalues)
    
    print("\nEigenvectors:")
    print(eigenvectors)
    
    if np.all(np.isreal(eigenvalues)):
        print("\n" + "-"*50)
        print("INTERPRETATION:")
        print("-"*50)
        print("Eigenvectors show directions that are only scaled")
        print("(not rotated) by the transformation.")
        print("Eigenvalues show the scaling factor along each direction.")
    else:
        print("\nNote: Complex eigenvalues indicate rotation is involved.")
    
    return fig

def plot_grid_transformation(T, title_text):
    """Plot grid transformation"""
    
    fig = plt.figure(figsize=(14, 6))
    
    # Create grid
    x = np.arange(-2, 2.5, 0.5)
    y = np.arange(-2, 2.5, 0.5)
    X, Y = np.meshgrid(x, y)
    
    # Flatten grid points
    grid_points = np.vstack([X.ravel(), Y.ravel()])
    
    # Transform grid
    transformed_grid = T @ grid_points
    X_t = transformed_grid[0, :].reshape(X.shape)
    Y_t = transformed_grid[1, :].reshape(Y.shape)
    
    # Plot original grid
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(grid_points[0, :], grid_points[1, :], 'b.', markersize=10)
    
    # Draw grid lines
    for i in range(X.shape[0]):
        ax1.plot(X[i, :], Y[i, :], 'b-', linewidth=0.5, alpha=0.6)
    for j in range(X.shape[1]):
        ax1.plot(X[:, j], Y[:, j], 'b-', linewidth=0.5, alpha=0.6)
    
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)
    ax1.set_aspect('equal')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Original Grid', fontsize=14, fontweight='bold')
    
    # Plot transformed grid
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(transformed_grid[0, :], transformed_grid[1, :], 'r.', markersize=10)
    
    # Draw transformed grid lines
    for i in range(X_t.shape[0]):
        ax2.plot(X_t[i, :], Y_t[i, :], 'r-', linewidth=0.5, alpha=0.6)
    for j in range(X_t.shape[1]):
        ax2.plot(X_t[:, j], Y_t[:, j], 'r-', linewidth=0.5, alpha=0.6)
    
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    ax2.set_aspect('equal')
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.set_title('Transformed Grid', fontsize=14, fontweight='bold')
    
    fig.suptitle('Grid Transformation Visualization', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

def run_demo():
    """Run demonstration of various transformations"""
    
    print("\nRunning demonstration of various transformations...")
    
    square = create_square()
    
    transformations = [
        {'name': 'Rotation 45°', 'matrix': rotation_matrix(45)},
        {'name': 'Scaling (2x, 0.5y)', 'matrix': scaling_matrix(2, 0.5)},
        {'name': 'Horizontal Shear', 'matrix': shearing_matrix(0.5, horizontal=True)},
        {'name': 'Reflection over y=x', 'matrix': reflection_matrix('xy')},
        {'name': 'Rotation 90°', 'matrix': rotation_matrix(90)},
        {'name': 'Compression', 'matrix': scaling_matrix(0.5, 0.5)}
    ]
    
    fig = plt.figure(figsize=(15, 10))
    
    for i, trans in enumerate(transformations):
        T = trans['matrix']
        transformed = T @ square
        det_T = np.linalg.det(T)
        
        ax = plt.subplot(2, 3, i+1)
        
        # Plot original
        ax.fill(square[0, :], square[1, :], 'b', alpha=0.2,
               edgecolor='b', linewidth=1.5, label='Original')
        
        # Plot transformed
        ax.fill(transformed[0, :], transformed[1, :], 'r', alpha=0.3,
               edgecolor='r', linewidth=2, label='Transformed')
        
        # Plot basis vectors
        ax.plot([0, 1], [0, 0], 'b--', linewidth=1, alpha=0.5)
        ax.plot([0, 0], [0, 1], 'b--', linewidth=1, alpha=0.5)
        
        tx = T @ np.array([1, 0])
        ty = T @ np.array([0, 1])
        ax.arrow(0, 0, tx[0], tx[1], head_width=0.1, head_length=0.1,
                fc='r', ec='r', linewidth=1.5)
        ax.arrow(0, 0, ty[0], ty[1], head_width=0.1, head_length=0.1,
                fc='g', ec='g', linewidth=1.5)
        
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{trans["name"]}\ndet = {det_T:.2f}', fontweight='bold')
        ax.legend(loc='best', fontsize=8)
    
    fig.suptitle('2D Linear Transformations - Demo Mode', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def main():
    """Main function"""
    
    print("="*50)
    print("2D LINEAR TRANSFORMATION VISUALIZER")
    print("="*50)
    print("\nChoose transformation type:")
    print("1. Rotation")
    print("2. Scaling")
    print("3. Shearing")
    print("4. Reflection")
    print("5. Custom Matrix")
    print("6. Multiple Transformations Demo")
    
    try:
        choice = int(input("\nEnter your choice (1-6): "))
    except ValueError:
        print("Invalid input! Please run again.")
        return
    
    shape = create_square()
    
    if choice == 1:  # Rotation
        angle = float(input("Enter rotation angle in degrees: "))
        T = rotation_matrix(angle)
        title_text = f'Rotation by {angle:.1f} degrees'
        
    elif choice == 2:  # Scaling
        sx = float(input("Enter x-scaling factor: "))
        sy = float(input("Enter y-scaling factor: "))
        T = scaling_matrix(sx, sy)
        title_text = f'Scaling (sx={sx:.2f}, sy={sy:.2f})'
        
    elif choice == 3:  # Shearing
        print("\nShearing options:")
        print("1. Horizontal shear")
        print("2. Vertical shear")
        shear_type = int(input("Choose (1 or 2): "))
        k = float(input("Enter shear factor: "))
        
        T = shearing_matrix(k, horizontal=(shear_type == 1))
        shear_dir = "Horizontal" if shear_type == 1 else "Vertical"
        title_text = f'{shear_dir} Shear (k={k:.2f})'
        
    elif choice == 4:  # Reflection
        print("\nReflection options:")
        print("1. Over x-axis")
        print("2. Over y-axis")
        print("3. Over y=x line")
        ref_type = int(input("Choose (1-3): "))
        
        axis_map = {1: 'x', 2: 'y', 3: 'xy'}
        axis = axis_map.get(ref_type, 'x')
        T = reflection_matrix(axis)
        title_text = f'Reflection over {"x-axis" if axis=="x" else "y-axis" if axis=="y" else "y=x"}'
        
    elif choice == 5:  # Custom Matrix
        print("\nEnter 2x2 transformation matrix:")
        T = np.zeros((2, 2))
        T[0, 0] = float(input("T[0,0] = "))
        T[0, 1] = float(input("T[0,1] = "))
        T[1, 0] = float(input("T[1,0] = "))
        T[1, 1] = float(input("T[1,1] = "))
        title_text = 'Custom Transformation'
        
    elif choice == 6:  # Demo
        run_demo()
        return
        
    else:
        print("Invalid choice!")
        return
    
    # Apply transformation
    transformed_shape = T @ shape
    
    # Visualize
    fig1 = plot_transformation(shape, transformed_shape, T, title_text)
    fig2 = plot_grid_transformation(T, title_text)
    
    plt.show()

if __name__ == "__main__":
    main()