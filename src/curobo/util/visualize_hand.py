import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import torch
import trimesh as tm
import transforms3d
import yourdfpy as urdf


@dataclass
class URDFVisualizerConfig:
    urdf_path: str
    collision_spheres_path: Optional[str] = None
    mesh_directory: Optional[str] = None
    joint_angles: Optional[Dict[str, float]] = None
    base_position: List[float] = None
    base_orientation: List[float] = None
    
    def __post_init__(self):
        if self.base_position is None:
            self.base_position = [0.0, 0.0, 0.0]
        if self.base_orientation is None:
            self.base_orientation = [0.0, 0.0, 0.0]


class URDFPenetrationVisualizer:
    def __init__(self, config: URDFVisualizerConfig, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        
        # Load URDF using yourdfpy
        self.urdf_model = urdf.URDF.load(config.urdf_path)
        self.urdf_path = Path(config.urdf_path)
        
        # Load collision spheres config to get link list
        self.collision_link_names = None
        if config.collision_spheres_path:
            self._load_collision_config()
        
        # Store link meshes and metadata
        self.link_data = {}
        self._load_link_meshes()
        
        # Set robot state
        self._set_robot_state()
    
    def _load_collision_config(self):
        """Load collision spheres config and extract link names."""
        with open(self.config.collision_spheres_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Get the list of collision link names
        self.collision_link_names = data.get('collision_link_names', [])
        
        # Get collision spheres if they exist
        self.collision_spheres_data = data.get('collision_spheres', {})
    
    def _should_include_link(self, link_name: str) -> bool:
        """Check if link should be included based on collision_link_names."""
        if self.collision_link_names is None:
            # If no collision config, include all links
            return True
        
        # Only include if in the collision_link_names list
        return link_name in self.collision_link_names
    
    def _load_mesh_from_file(self, mesh_geometry, origin=None):
        """Load a mesh from file given a mesh geometry object."""
        if mesh_geometry.filename is None:
            return None
        
        # Resolve mesh path relative to URDF
        mesh_filename = str(mesh_geometry.filename)
        
        # Remove package:// prefix if present
        if 'package://' in mesh_filename:
            mesh_filename = mesh_filename.replace('package://', '')
        
        mesh_path = Path(mesh_filename)
        if not mesh_path.is_absolute():
            # Try relative to URDF directory
            mesh_path = self.urdf_path.parent / mesh_filename
        
        try:
            mesh = tm.load(str(mesh_path))
            
            # Apply scale if specified
            if mesh_geometry.scale is not None:
                scale_matrix = np.eye(4)
                scale_matrix[:3, :3] *= np.array(mesh_geometry.scale)
                mesh.apply_transform(scale_matrix)
            
            # Apply origin transform if specified
            if origin is not None:
                mesh.apply_transform(origin)
            
            return mesh
        except Exception as e:
            print(f"Warning: Could not load mesh from {mesh_path}: {e}")
            return None
    
    def _load_link_meshes(self):
        """Load mesh data for each link using yourdfpy."""
        for link_name, link in self.urdf_model.link_map.items():
            # Check if we should include this link
            if not self._should_include_link(link_name):
                continue
            
            if not link.collisions:
                print(f"Warning: Link {link_name} in collision_link_names has no collision geometry")
                continue
            
            self.link_data[link_name] = {}
            
            # Get collision mesh from yourdfpy
            collision_meshes = []
            for collision in link.collisions:
                mesh = None
                if collision.geometry.mesh is not None:
                    mesh = self._load_mesh_from_file(collision.geometry.mesh, collision.origin)
                elif collision.geometry.box is not None:
                    box = collision.geometry.box
                    mesh = tm.creation.box(extents=box.size)
                    if collision.origin is not None:
                        mesh.apply_transform(collision.origin)
                elif collision.geometry.sphere is not None:
                    sphere = collision.geometry.sphere
                    mesh = tm.primitives.Sphere(radius=sphere.radius)
                    if collision.origin is not None:
                        mesh.apply_transform(collision.origin)
                elif collision.geometry.cylinder is not None:
                    cylinder = collision.geometry.cylinder
                    mesh = tm.primitives.Cylinder(
                        radius=cylinder.radius,
                        height=cylinder.length
                    )
                    if collision.origin is not None:
                        mesh.apply_transform(collision.origin)
                
                if mesh is not None:
                    collision_meshes.append(mesh)
            
            if collision_meshes:
                combined_mesh = tm.util.concatenate(collision_meshes)
                self.link_data[link_name]["vertices"] = torch.tensor(
                    combined_mesh.vertices, dtype=torch.float, device=self.device
                )
                self.link_data[link_name]["faces"] = torch.tensor(
                    combined_mesh.faces, dtype=torch.long, device=self.device
                )
            
            # Get visual meshes if available
            visual_meshes = []
            for visual in link.visuals:
                if visual.geometry.mesh is not None:
                    mesh = self._load_mesh_from_file(visual.geometry.mesh, visual.origin)
                    if mesh is not None:
                        visual_meshes.append(mesh)
            
            if visual_meshes:
                combined_visual = tm.util.concatenate(visual_meshes)
                self.link_data[link_name]["visual_vertices"] = torch.tensor(
                    combined_visual.vertices, dtype=torch.float, device=self.device
                )
                self.link_data[link_name]["visual_faces"] = torch.tensor(
                    combined_visual.faces, dtype=torch.long, device=self.device
                )
    
    def _load_collision_spheres(self):
        """Load collision spheres from the config data."""
        if not hasattr(self, 'collision_spheres_data'):
            return
        
        for link_name, spheres in self.collision_spheres_data.items():
            if link_name not in self.link_data:
                if self._should_include_link(link_name):
                    print(f"Warning: Link {link_name} from collision_spheres not found in URDF")
                continue
            
            sphere_data = []
            for sphere in spheres:
                center = torch.tensor(
                    sphere['center'], 
                    dtype=torch.float, 
                    device=self.device
                )
                radius = float(sphere['radius'])
                sphere_data.append((center, radius))
            
            self.link_data[link_name]["collision_spheres"] = sphere_data
    
    def _set_robot_state(self):
        """Set robot joint angles and base pose."""
        if self.config.joint_angles is None:
            cfg = {}
        else:
            cfg = self.config.joint_angles
        
        # Compute forward kinematics using yourdfpy's scene
        # Create a configuration dict with all joints (use 0 for unspecified)
        full_cfg = {}
        for joint_name, joint in self.urdf_model.joint_map.items():
            if joint.type != 'fixed':
                full_cfg[joint_name] = cfg.get(joint_name, 0.0)
        
        # Get link transforms using the scene
        self.fk_result = {}
        scene = self.urdf_model.scene
        
        # Update scene with configuration
        for joint_name, angle in full_cfg.items():
            if joint_name in scene.graph.transforms.node_data:
                # This sets the joint angle in the scene graph
                pass  # yourdfpy handles this internally when we call get_transform
        
        # Get transforms for each link
        for link_name in self.urdf_model.link_map.keys():
            try:
                # Get transform from scene - this is relative to base_link
                transform = scene.graph.get(frame_to=link_name, frame_from=scene.graph.base_frame)
                if transform is not None:
                    self.fk_result[link_name] = transform[0]  # get_transform returns (matrix, metadata)
                else:
                    self.fk_result[link_name] = np.eye(4)
            except:
                # If we can't get the transform, use identity
                self.fk_result[link_name] = np.eye(4)
        
        # Set base transformation
        self.base_translation = torch.tensor(
            self.config.base_position, dtype=torch.float, device=self.device
        )
        self.base_rotation = torch.tensor(
            transforms3d.euler.euler2mat(*self.config.base_orientation),
            dtype=torch.float,
            device=self.device
        )
        
        # Now load collision spheres (needs FK to be computed)
        if self.config.collision_spheres_path:
            self._load_collision_spheres()
    
    def get_transformed_points(self, link_name: str, points: torch.Tensor) -> np.ndarray:
        """Transform points from link frame to world frame."""
        # Get link transform from FK
        if link_name in self.fk_result:
            link_transform = self.fk_result[link_name]
        else:
            link_transform = np.eye(4)
        
        # Convert points to numpy and homogeneous coordinates
        points_np = points.cpu().numpy()
        points_homo = np.hstack([points_np, np.ones((points_np.shape[0], 1))])
        
        # Apply link transform
        points_link = (link_transform @ points_homo.T).T[:, :3]
        
        # Apply base transform
        base_rot = self.base_rotation.cpu().numpy()
        base_trans = self.base_translation.cpu().numpy()
        points_world = points_link @ base_rot.T + base_trans
        
        return points_world
    
    def visualize(
        self,
        opacity: float = 0.5,
        mesh_color: tuple = (173, 216, 230, 128),  # lightblue with alpha
        sphere_color: tuple = (222, 184, 135, 128),  # burlywood with alpha
        sphere_center_color: tuple = (255, 0, 0, 255),  # red
        use_visual_mesh: bool = False,
        show_sphere_centers: bool = True,
        show_collision_spheres: bool = True,
        output_html: str = "urdf_collision_spheres.html"
    ) -> tm.Scene:
        """
        Create visualization and export to HTML.
        
        Colors are RGBA tuples (0-255).
        """
        scene = tm.Scene()
        
        # Convert opacity to alpha (0-255)
        mesh_alpha = int(opacity * 255)
        
        # Add link meshes
        for link_name, link_data in self.link_data.items():
            if use_visual_mesh and "visual_vertices" in link_data:
                vertices = link_data["visual_vertices"]
                faces = link_data["visual_faces"]
            else:
                if "vertices" not in link_data:
                    continue
                vertices = link_data["vertices"]
                faces = link_data["faces"]
            
            # Transform to world frame
            vertices_world = self.get_transformed_points(link_name, vertices)
            
            # Create mesh with color
            mesh = tm.Trimesh(
                vertices=vertices_world,
                faces=faces.cpu().numpy()
            )
            
            # Set color with opacity
            color_with_alpha = list(mesh_color[:3]) + [mesh_alpha]
            mesh.visual.face_colors = color_with_alpha
            
            scene.add_geometry(mesh, node_name=f"link_{link_name}")
        
        # Add collision spheres and centers
        if show_collision_spheres or show_sphere_centers:
            for link_name, link_data in self.link_data.items():
                if "collision_spheres" not in link_data:
                    continue
                
                for i, (center, radius) in enumerate(link_data["collision_spheres"]):
                    center_world = self.get_transformed_points(link_name, center.unsqueeze(0))[0]
                    
                    # Add sphere
                    if show_collision_spheres:
                        sphere_mesh = tm.primitives.Sphere(
                            radius=radius,
                            center=center_world,
                            subdivisions=3  # Lower for better performance
                        )
                        sphere_mesh.visual.face_colors = sphere_color
                        scene.add_geometry(sphere_mesh, node_name=f"sphere_{link_name}_{i}")
                    
                    # Add center point as tiny sphere for visibility
                    if show_sphere_centers:
                        center_sphere = tm.primitives.Sphere(
                            radius=radius * 0.1,  # 10% of sphere radius
                            center=center_world,
                            subdivisions=1
                        )
                        center_sphere.visual.face_colors = sphere_center_color
                        scene.add_geometry(center_sphere, node_name=f"center_{link_name}_{i}")
        
        # Export to GLB (can be viewed in browser or 3D viewers)
        if output_html:
            if output_html.endswith('.html'):
                # Create HTML viewer with embedded GLB
                glb_file = output_html.replace('.html', '.glb')
                scene.export(glb_file)
                self._create_html_viewer(glb_file, output_html)
                print(f"✓ Visualization saved to {output_html}")
                print(f"  Open in browser to view (requires serving via HTTP)")
                print(f"  Or download GLB: {glb_file}")
            else:
                # Just export GLB
                scene.export(output_html)
                print(f"✓ Visualization saved to {output_html}")
                print(f"  Download and open in browser (drag & drop)")
                print(f"  Or use online viewers like https://gltf-viewer.donmccurdy.com/")
            
            print(f"  Visualized {len(self.link_data)} links")
        
        return scene
    
    def _create_html_viewer(self, glb_file: str, html_file: str):
        """Create an HTML viewer for the GLB file using Three.js."""
        glb_filename = Path(glb_file).name
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>URDF Visualization</title>
    <style>
        body {{ margin: 0; overflow: hidden; }}
        canvas {{ display: block; }}
        #info {{
            position: absolute;
            top: 10px;
            left: 10px;
            color: white;
            background: rgba(0,0,0,0.7);
            padding: 10px;
            border-radius: 5px;
            font-family: monospace;
        }}
    </style>
</head>
<body>
    <div id="info">
        <strong>URDF Collision Spheres Visualization</strong><br>
        Mouse: Rotate | Scroll: Zoom | Right-click: Pan
    </div>
    <script type="importmap">
    {{
        "imports": {{
            "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
            "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
        }}
    }}
    </script>
    <script type="module">
        import * as THREE from 'three';
        import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
        import {{ GLTFLoader }} from 'three/addons/loaders/GLTFLoader.js';

        // Scene setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1a1a);
        
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.001, 1000);
        camera.position.set(0.3, 0.3, 0.3);
        
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);
        
        // Controls
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        
        // Lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        
        const directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight1.position.set(1, 1, 1);
        scene.add(directionalLight1);
        
        const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
        directionalLight2.position.set(-1, -1, -1);
        scene.add(directionalLight2);
        
        // Grid
        const gridHelper = new THREE.GridHelper(1, 10);
        scene.add(gridHelper);
        
        // Axes helper
        const axesHelper = new THREE.AxesHelper(0.1);
        scene.add(axesHelper);
        
        // Load GLB
        const loader = new GLTFLoader();
        loader.load('{glb_filename}', function(gltf) {{
            scene.add(gltf.scene);
            
            // Auto-fit camera
            const box = new THREE.Box3().setFromObject(gltf.scene);
            const center = box.getCenter(new THREE.Vector3());
            const size = box.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);
            const fov = camera.fov * (Math.PI / 180);
            let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
            cameraZ *= 1.5; // Add some padding
            
            camera.position.set(center.x + cameraZ * 0.5, center.y + cameraZ * 0.5, center.z + cameraZ);
            camera.lookAt(center);
            controls.target.copy(center);
            controls.update();
        }}, undefined, function(error) {{
            console.error('Error loading GLB:', error);
        }});
        
        // Animation loop
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        
        // Handle window resize
        window.addEventListener('resize', function() {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
        
        animate();
    </script>
</body>
</html>"""
        
        with open(html_file, 'w') as f:
            f.write(html_content)
    
    def print_link_names(self, show_all: bool = False):
        """
        Print link names.
        
        Args:
            show_all: If True, show all links in URDF. If False, show only filtered links.
        """
        if show_all:
            print("All available links in URDF:")
            for link_name in self.urdf_model.link_map.keys():
                included = "✓" if link_name in self.link_data else "✗"
                in_collision_list = "C" if (self.collision_link_names and link_name in self.collision_link_names) else " "
                print(f"  [{included}][{in_collision_list}] {link_name}")
            print("\nLegend: ✓=loaded, C=in collision_link_names")
        else:
            print(f"Loaded links ({len(self.link_data)}):")
            for link_name in self.link_data.keys():
                print(f"  - {link_name}")
    
    def print_joint_names(self, show_all: bool = False):
        """
        Print joint names.
        
        Args:
            show_all: If True, show all joints in URDF. If False, show only joints for filtered links.
        """
        relevant_links = set(self.link_data.keys())
        
        print("\nAvailable joints:")
        for joint_name, joint in self.urdf_model.joint_map.items():
            if joint.type == 'fixed':
                continue
            
            # Check if this joint is relevant to our filtered links
            parent_relevant = joint.parent in relevant_links if not show_all else True
            child_relevant = joint.child in relevant_links if not show_all else True
            is_relevant = parent_relevant or child_relevant
            
            if show_all or is_relevant:
                marker = "✓" if is_relevant else "✗"
                print(f"  [{marker}] {joint_name} (type: {joint.type}, parent: {joint.parent}, child: {joint.child})")
    
    def check_missing_links(self):
        """Check if any links in collision_link_names are missing from URDF."""
        if self.collision_link_names is None:
            print("No collision_link_names specified")
            return
        
        missing_links = []
        for link_name in self.collision_link_names:
            if link_name not in self.urdf_model.link_map:
                missing_links.append(link_name)
        
        if missing_links:
            print(f"⚠️  WARNING: {len(missing_links)} links in collision_link_names not found in URDF:")
            for link_name in missing_links:
                print(f"  - {link_name}")
        else:
            print(f"✓ All {len(self.collision_link_names)} collision links found in URDF")


# Example usage
if __name__ == "__main__":
    config = URDFVisualizerConfig(
        urdf_path="/home/kaidikang/BODex/src/curobo/content/assets/robot/vega_1/vega.urdf",
        collision_spheres_path="/home/kaidikang/BODex/src/curobo/content/configs/robot/spheres/vega1_dexmate.yml",  # Contains collision_link_names
        joint_angles={
            # Only specify hand joint angles
            
            # Inspire left
            # "L_thumb_proximal_yaw_joint": 0.0,  # Thumb opposition
            # "L_thumb_proximal_pitch_joint": 0.0,  # Thumb flexion
            # "L_index_proximal_joint": 0.0,  # Index flexion
            # "L_middle_proximal_joint": 0.0,  # Middle flexion
            # "L_ring_proximal_joint": 0.0,  # Ring flexion
            # "L_pinky_proximal_joint": 0.0,  # Pinky flexion
            
            # Inspire right
            # "R_thumb_proximal_yaw_joint": 0.0,  # Thumb opposition
            # "R_thumb_proximal_pitch_joint": 0.0,  # Thumb flexion
            # "R_index_proximal_joint": 0.0,  # Index flexion
            # "R_middle_proximal_joint": 0.0,  # Middle flexion
            # "R_ring_proximal_joint": 0.0,  # Ring flexion
            # "R_pinky_proximal_joint": 0.0,  # Pinky flexion

            # Dexmate right
            # "R_th_j0": 0.0,          # Thumb opposition/yaw
            # "R_th_j1": 0.0,          # Thumb flexion/pitch
            # "R_ff_j1": 0.0,          # Index flexion
            # "R_mf_j1": 0.0,          # Middle flexion
            # "R_rf_j1": 0.0,          # Ring flexion
            # "R_lf_j1": 0.0,          # Pinky flexion

            # Dexmate left
            "L_th_j0": 0.0,          # Thumb opposition/yaw
            "L_th_j1": 0.0,          # Thumb flexion/pitch
            "L_ff_j1": 0.0,          # Index flexion
            "L_mf_j1": 0.0,          # Middle flexion
            "L_rf_j1": 0.0,          # Ring flexion
            "L_lf_j1": 0.0,          # Pinky flexion
        },
        base_position=[0.0, 0.0, 0.0],
        base_orientation=[0.0, 0.0, 0.0],
    )
    
    visualizer = URDFPenetrationVisualizer(config)
    
    # Check if all links in yaml are found in URDF
    visualizer.check_missing_links()
    
    # Print what got loaded
    visualizer.print_link_names(show_all=False)
    visualizer.print_joint_names(show_all=False)
    
    # See all links in URDF (optional)
    # visualizer.print_link_names(show_all=True)
    
    # Create visualization
    scene = visualizer.visualize(
        opacity=0.5,
        use_visual_mesh=False,
        show_sphere_centers=True,
        show_collision_spheres=True,
        output_html="right_hand_visualization.glb"  # Changed to .glb
    )