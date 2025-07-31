import numpy as np
import matplotlib.pyplot as plt
import os

def generate_revolution_solid_stl(points, filename="revolution_solid.stl", num_segments=36, visualize=False, save_plot_path=None):
    """
    Generates a solid of revolution around Z-axis from 3D points and saves as STL.
    
    Parameters:
    -----------
    points : array_like
        Points in format [(x1,y1,z1), (x2,y2,z2), ...] (minimum 3 points)
    filename : str
        STL file name
    num_segments : int
        Revolution segments (more = smoother)
    visualize : bool
        Show plot
    save_plot_path : str
        Directory to save profile plot
    
    Returns:
    --------
    dict : Solid information
    """
    
    points = np.array(points)
    
    if points.shape[1] != 3:
        raise ValueError("Points must have 3 coordinates (x, y, z)")
    if len(points) < 3:
        raise ValueError("Minimum 3 points required")
    
    def _generate_closed_profile(pts):
        """Generate closed profile from points"""
        # Sort by Z coordinate
        sorted_indices = np.argsort(pts[:, 2])
        sorted_points = pts[sorted_indices]
        
        # Calculate radius from Z-axis: r = sqrt(x² + y²)
        radii = np.sqrt(sorted_points[:, 0]**2 + sorted_points[:, 1]**2)
        heights = sorted_points[:, 2]
        
        profile = np.column_stack([radii, heights])
        
        # Close profile if needed
        if not np.allclose(profile[0], profile[-1], atol=1e-6):
            profile = np.vstack([profile, profile[0]])
        
        return profile
    
    def _create_revolution_solid(profile, num_seg):
        """Create revolution solid vertices and triangles"""
        vertices = []
        triangles = []
        
        angles = np.linspace(0, 2*np.pi, num_seg + 1)
        
        # Generate vertices
        for i, angle in enumerate(angles[:-1]):
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
                v1 = i * num_profile_points + j
                v2 = i * num_profile_points + (j + 1)
                v3 = i_next * num_profile_points + j
                v4 = i_next * num_profile_points + (j + 1)
                
                triangles.append([v1, v2, v3])
                triangles.append([v2, v4, v3])
        
        return vertices, np.array(triangles)
    
    def _calculate_normal(v1, v2, v3):
        """Calculate triangle normal"""
        edge1 = v2 - v1
        edge2 = v3 - v1
        normal = np.cross(edge1, edge2)
        norm_magnitude = np.linalg.norm(normal)
        if norm_magnitude > 1e-10:
            return normal / norm_magnitude
        else:
            return np.array([0, 0, 1])
    
    def _save_stl(vertices, triangles, fname):
        """Save solid in STL format"""
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
        """Visualize 2D profile"""
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        
        ax.plot(profile[:, 0], profile[:, 1], 'bo-', linewidth=2, markersize=6)
        
        ax.set_xlabel('Radius (distance from Z-axis) [m]')
        ax.set_ylabel('Height (Z) [m]')
        ax.set_title(f'2D Profile for Z-axis Revolution ({len(points)} points)')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        plt.tight_layout()
        
        if save_path:
            plot_filename = os.path.join(save_path, "profile_plot.png")
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        
        if visualize:
            plt.show()
        else:
            plt.close()
    
    # Generate profile and solid
    profile = _generate_closed_profile(points)
    vertices, triangles = _create_revolution_solid(profile, num_segments)
    
    # Save STL
    _save_stl(vertices, triangles, filename)
    
    # Visualize if requested
    if visualize or save_plot_path:
        _visualize_profile_and_solid(profile, vertices, triangles, save_path=save_plot_path)
    
    # Get file size
    file_size = os.path.getsize(filename) if os.path.exists(filename) else 0
    
    return {
        'filename': filename,
        'num_vertices': len(vertices),
        'num_triangles': len(triangles),
        'file_size_bytes': file_size,
        'num_original_points': len(points),
        'num_profile_points': len(profile),
        'vertices': vertices,
        'triangles': triangles,
        'profile': profile,
        'original_points': points
    }


if __name__ == "__main__":
    # Example usage
    P = np.array([
        [3, 0, -2],   # P1
        [5, 0, -2],   # P2
        [10, 0, -1],  # P3
        [10, 0, 3],   # P4
        [3, 0, 3]     # P5
    ])
    
    result = generate_revolution_solid_stl(
        points=P,
        filename="test_solid.stl",
        num_segments=36,
        visualize=True
    )
    
    print(f"STL created: {result['filename']}")
    print(f"Vertices: {result['num_vertices']}")
    print(f"Triangles: {result['num_triangles']}")
