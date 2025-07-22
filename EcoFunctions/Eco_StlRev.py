import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def generate_revolution_solid_stl(points, filename="revolution_solid.stl", num_segments=36, visualize=False, save_plot_path=None):
    """
    Generates a solid of revolution around Z-axis from 3D points and saves it as an STL file.
    
    Parameters:
    -----------
    points : list or np.array
        List of points in format [(x1,y1,z1), (x2,y2,z2), ...] or numpy array Nx3
    filename : str
        Name of the STL file to generate
    num_segments : int
        Number of segments for revolution (more segments = smoother surface)
    visualize : bool
        If True, displays plots of the profile and solid
    save_plot_path : str, optional
        Directory path where to save the profile plot. If None, saves in current directory.
    
    Returns:
    --------
    dict : Information about the generated solid
    """
    
    # Convert points to numpy array if necessary
    if isinstance(points, list):
        points = np.array(points)
    
    def _generate_closed_profile(pts):
        """Generate a closed profile from the given points"""
        # Sort points by Z coordinate (height)
        sorted_points = pts[np.argsort(pts[:, 2])]
        
        # Create closed profile using distance from Z-axis as radius and Z as height
        # Calculate radius from Z-axis: r = sqrt(x² + y²)
        radii = np.sqrt(sorted_points[:, 0]**2 + sorted_points[:, 1]**2)
        heights = sorted_points[:, 2]
        
        profile = np.column_stack([radii, heights])  # [radius, height]
        
        # Close the profile if necessary
        if not np.allclose(profile[0], profile[-1]):
            profile = np.vstack([profile, profile[0]])
        
        return profile
    
    def _create_revolution_solid(profile, num_seg):
        """Create a solid of revolution around Z-axis from the given profile"""
        vertices = []
        triangles = []
        
        # Generate angles for revolution around Z-axis
        angles = np.linspace(0, 2*np.pi, num_seg + 1)
        
        # Generate vertices
        for i, angle in enumerate(angles[:-1]):  # Exclude the last duplicate point
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            
            for j, point in enumerate(profile):
                radius, height = point
                x = radius * cos_a
                y = radius * sin_a
                z = height
                vertices.append([x, y, z])
        
        vertices = np.array(vertices)
        
        # Generate triangles
        num_profile_points = len(profile)
        
        for i in range(num_seg):
            i_next = (i + 1) % num_seg
            
            for j in range(num_profile_points - 1):
                # Vertex indices
                v1 = i * num_profile_points + j
                v2 = i * num_profile_points + (j + 1)
                v3 = i_next * num_profile_points + j
                v4 = i_next * num_profile_points + (j + 1)
                
                # Create two triangles per quadrilateral
                triangles.append([v1, v2, v3])
                triangles.append([v2, v4, v3])
        
        return vertices, np.array(triangles)
    
    def _calculate_normal(v1, v2, v3):
        """Calculate the normal of a triangle"""
        edge1 = v2 - v1
        edge2 = v3 - v1
        normal = np.cross(edge1, edge2)
        norm_magnitude = np.linalg.norm(normal)
        if norm_magnitude > 1e-10:
            return normal / norm_magnitude
        else:
            return np.array([0, 0, 1])  # Default normal
    
    def _save_stl(vertices, triangles, fname):
        """Save the solid in STL format"""
        with open(fname, 'w') as f:
            f.write("solid revolution_solid\n")
            
            for tri in triangles:
                v1, v2, v3 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
                normal = _calculate_normal(v1, v2, v3)
                
                f.write(f"  facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
                f.write("    outer loop\n")
                f.write(f"      vertex {v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}\n")
                f.write(f"      vertex {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}\n")
                f.write(f"      vertex {v3[0]:.6f} {v3[1]:.6f} {v3[2]:.6f}\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")
            
            f.write("endsolid revolution_solid\n")
    
    def _visualize_profile_and_solid(profile, vertices, triangles, save_path=None):
        """Visualize the 2D profile and optionally save it"""
        fig = plt.figure(figsize=(8, 6))
        
        # 2D Profile only
        ax = fig.add_subplot(111)
        ax.plot(profile[:, 0], profile[:, 1], 'b-', linewidth=2, marker='o')
        ax.set_xlabel('Radius (distancia desde eje Z)')
        ax.set_ylabel('Altura (Z)')
        ax.set_title('Perfil 2D para Revolución en Z')
        ax.grid(True)
        ax.axis('equal')
        
        plt.tight_layout()
        
        # Save the plot if save_path is provided
        if save_path:
            plot_filename = os.path.join(save_path, "profile_plot.png")
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Gráfico del perfil guardado en: {plot_filename}")
        
        if visualize:
            plt.show()
        else:
            plt.close()  # Close the figure to free memory if not showing
    
    # Validate input
    if points.shape[1] != 3:
        raise ValueError("Points must have 3 coordinates (x, y, z)")
    
    # Generate closed profile
    profile = _generate_closed_profile(points)
    
    # Create revolution solid
    vertices, triangles = _create_revolution_solid(profile, num_segments)
    
    # Save as STL
    _save_stl(vertices, triangles, filename)
    
    # Visualize and/or save plot
    _visualize_profile_and_solid(profile, vertices, triangles, save_path=save_plot_path)
    
    # Get file information
    file_size = os.path.getsize(filename) if os.path.exists(filename) else 0
    
    # Return information
    return {
        'filename': filename,
        'num_vertices': len(vertices),
        'num_triangles': len(triangles),
        'file_size_bytes': file_size,
        'vertices': vertices,
        'triangles': triangles,
        'profile': profile,
        'original_points': points
    }


# Example usage
if __name__ == "__main__":
    # Points given as vectors
    P = np.array([
        [3, 0, -2],   # P1
        [5, 0, -2],   # P2
        [10, 0, -1],  # P3
        [10, 0, 3],   # P4
        [3, 0, 3]     # P5
    ])
    
    print("Original points:")
    for i, point in enumerate(P, 1):
        print(f"P{i}: [{point[0]}, {point[1]}, {point[2]}]")
    
    print("\nRadios calculados desde el eje Z:")
    for i, point in enumerate(P, 1):
        radius = np.sqrt(point[0]**2 + point[1]**2)
        print(f"P{i}: radio = {radius:.2f}, altura = {point[2]}")
    
    # Generate revolution solid
    result = generate_revolution_solid_stl(
        points=P,
        filename="my_solid_z_axis.stl",
        num_segments=36,
        visualize=True
    )
    
    print(f"\nResult:")
    print(f"- File: {result['filename']}")
    print(f"- Vertices: {result['num_vertices']:,}")
    print(f"- Triangles: {result['num_triangles']:,}")
    print(f"- Size: {result['file_size_bytes']:,} bytes")
    print(f"- STL file created successfully!")
    print(f"- Revolución generada alrededor del eje Z")