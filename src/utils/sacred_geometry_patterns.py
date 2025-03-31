import numpy as np
import math
from .sacred_geometry_core import SacredGeometryCore

class SacredGeometryPatterns(SacredGeometryCore):
    """
    Extension of SacredGeometryCore that provides specific patterns and processing 
    capabilities for sacred geometry in audio and frequency transformation.
    
    This class implements advanced pattern generation algorithms and transformations
    based on sacred geometry principles including:
    - Flower of Life
    - Metatron's Cube
    - Fibonacci Spiral
    - Sri Yantra
    - Quantum Coherence Patterns
    - Golden Ratio Harmonics
    """
    
    # Pattern types available for generation
    PATTERN_TYPES = {
        "flower_of_life": "Flower of Life pattern with overlapping circles",
        "metatrons_cube": "Metatron's Cube derived from Flower of Life",
        "fibonacci_spiral": "Fibonacci spiral based on golden ratio",
        "sri_yantra": "Sri Yantra sacred geometry pattern",
        "vesica_piscis": "Vesica Piscis - overlapping circles",
        "seed_of_life": "Seed of Life - 7 circle pattern",
        "tree_of_life": "Kabbalistic Tree of Life pattern",
        "merkaba": "Merkaba - 3D star tetrahedron",
        "torus": "Toroidal energy flow pattern",
        "golden_spiral": "Golden spiral based on phi ratio",
        "quantum_field": "Quantum field coherence pattern",
        "sacred_circuits": "Sacred circuits for enhanced flow",
        "phi_grid": "Phi-based grid pattern",
        "harmonic_lattice": "Harmonic frequency lattice",
        "chakra_mandala": "Seven chakra mandala system",
        "platonic_solids": "Five platonic solids patterns",
        "cosmometric": "Cosmometric harmony pattern",
        "unified_field": "Unified field resonance pattern",
        "holofractal": "Holofractal recursive pattern",
        "consciousness_grid": "Consciousness alignment grid",
        "light_language": "Light language geometric symbols",
        "sonic_mandala": "Sound-based mandala pattern",
        "quantum_lotus": "Quantum lotus unfolding pattern",
        "crystalline_grid": "Crystalline consciousness grid"
    }
    
    def __init__(self, dimensions=2, resolution=512, phi=1.618033988749895):
        """
        Initialize the SacredGeometryPatterns object.
        
        Args:
            dimensions: Number of dimensions for pattern generation (default: 2)
            resolution: Resolution of the generated patterns (default: 512)
            phi: Golden ratio value for pattern generation (default: 1.618033988749895)
        """
        super().__init__(dimensions, resolution)
        self.phi = phi
        self.pattern_cache = {}
        self.frequency_matrix = self._initialize_frequency_matrix()
        
    def _initialize_frequency_matrix(self):
        """Initialize the frequency matrix for pattern-based frequency modulation"""
        matrix = np.zeros((self.resolution, self.resolution), dtype=np.complex128)
        
        # Create phi-based frequency relationships
        for i in range(self.resolution):
            for j in range(self.resolution):
                distance = math.sqrt((i - self.resolution/2)**2 + (j - self.resolution/2)**2)
                angle = math.atan2(j - self.resolution/2, i - self.resolution/2)
                
                # Generate phi-based pattern
                phi_factor = (distance / self.resolution) * self.phi
                phase = angle + (phi_factor * 2 * math.pi)
                
                # Use complex numbers to represent phase and amplitude
                matrix[i, j] = complex(math.cos(phase), math.sin(phase)) * math.exp(-distance/self.resolution)
                
        return matrix
    
    def generate_pattern(self, pattern_type, intensity=1.0, phase_shift=0.0, seed=None):
        """
        Generate a specified sacred geometry pattern.
        
        Args:
            pattern_type: Type of pattern to generate (from PATTERN_TYPES)
            intensity: Intensity of the pattern (0.0-1.0)
            phase_shift: Phase shift to apply to the pattern
            seed: Random seed for reproducible patterns
            
        Returns:
            2D numpy array containing the generated pattern
        """
        # Check if pattern is in cache to improve performance
        cache_key = f"{pattern_type}_{intensity}_{phase_shift}_{seed}"
        if cache_key in self.pattern_cache:
            return self.pattern_cache[cache_key]
            
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            
        # Validate pattern type
        if pattern_type not in self.PATTERN_TYPES:
            raise ValueError(f"Pattern type '{pattern_type}' not recognized. Available patterns: {list(self.PATTERN_TYPES.keys())}")
            
        # Generate the requested pattern
        if pattern_type == "flower_of_life":
            pattern = self._generate_flower_of_life(intensity, phase_shift)
        elif pattern_type == "metatrons_cube":
            pattern = self._generate_metatrons_cube(intensity, phase_shift)
        elif pattern_type == "fibonacci_spiral":
            pattern = self._generate_fibonacci_spiral(intensity, phase_shift)
        elif pattern_type == "sri_yantra":
            pattern = self._generate_sri_yantra(intensity, phase_shift)
        elif pattern_type == "quantum_field":
            pattern = self._generate_quantum_field(intensity, phase_shift)
        elif pattern_type == "golden_spiral":
            pattern = self._generate_golden_spiral(intensity, phase_shift)
        elif pattern_type == "consciousness_grid":
            pattern = self._generate_consciousness_grid(intensity, phase_shift)
        else:
            # For other patterns, use the base method with modifications
            pattern = self._generate_generic_pattern(pattern_type, intensity, phase_shift)
            
        # Cache the generated pattern
        self.pattern_cache[cache_key] = pattern
        return pattern
        
    def _generate_flower_of_life(self, intensity=1.0, phase_shift=0.0):
        """Generate the Flower of Life pattern"""
        pattern = np.zeros((self.resolution, self.resolution), dtype=np.float64)
        center = self.resolution // 2
        radius = self.resolution // 6
        
        # Generate the seven main circles
        positions = [
            (center, center),  # Center circle
            (center, center - radius),  # Top
            (center + int(radius * 0.866), center - int(radius * 0.5)),  # Top right
            (center + int(radius * 0.866), center + int(radius * 0.5)),  # Bottom right
            (center, center + radius),  # Bottom
            (center - int(radius * 0.866), center + int(radius * 0.5)),  # Bottom left
            (center - int(radius * 0.866), center - int(radius * 0.5)),  # Top left
        ]
        
        # Draw the circles
        for cx, cy in positions:
            for i in range(self.resolution):
                for j in range(self.resolution):
                    dist = math.sqrt((i - cx)**2 + (j - cy)**2)
                    if dist <= radius:
                        angle = math.atan2(j - cy, i - cx) + phase_shift
                        value = 0.5 + 0.5 * math.cos(angle * 6)  # Create 6-fold symmetry
                        pattern[i, j] = max(pattern[i, j], value * intensity)
        
        return pattern
        
    def _generate_metatrons_cube(self, intensity=1.0, phase_shift=0.0):
        """Generate Metatron's Cube pattern"""
        pattern = np.zeros((self.resolution, self.resolution), dtype=np.float64)
        center = self.resolution // 2
        radius = self.resolution // 4
        
        # First generate the Flower of Life as the base
        flower = self._generate_flower_of_life(intensity * 0.5, phase_shift)
        pattern += flower
        
        # Add the lines connecting the vertices
        vertices = []
        for i in range(12):
            angle = i * (2 * math.pi / 12) + phase_shift
            x = center + int(radius * math.cos(angle))
            y = center + int(radius * math.sin(angle))
            vertices.append((x, y))
            
        # Connect all vertices to form Metatron's Cube
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                self._draw_line(pattern, vertices[i], vertices[j], intensity)
                
        return pattern
        
    def _generate_fibonacci_spiral(self, intensity=1.0, phase_shift=0.0):
        """Generate a Fibonacci spiral pattern based on the golden ratio"""
        pattern = np.zeros((self.resolution, self.resolution), dtype=np.float64)
        center = self.resolution // 2
        max_radius = self.resolution // 2 - 10
        
        # Draw the spiral
        theta = np.linspace(0, 8 * np.pi, 1000)  # 8 full revolutions
        radius = max_radius * (1 - np.exp(-0.3 * theta))
        
        for t, r in zip(theta, radius):
            angle = t + phase_shift
            x = center + int(r * math.cos(angle))
            y = center + int(r * math.sin(angle))
            
            if 0 <= x < self.resolution and 0 <= y < self.resolution:
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        xi, yi = x + dx, y + dy
                        if 0 <= xi < self.resolution and 0 <= yi < self.resolution:
                            dist = math.sqrt(dx**2 + dy**2)
                            if dist < 2:
                                pattern[xi, yi] = intensity * (1 - dist/2)
        
        return pattern
        
    def _generate_sri_yantra(self, intensity=1.0, phase_shift=0.0):
        """Generate Sri Yantra sacred geometry pattern"""
        pattern = np.zeros((self.resolution, self.resolution), dtype=np.float64)
        center = self.resolution // 2
        outer_radius = int(self.resolution * 0.45)
        
        # Draw the outer circle
        for i in range(self.resolution):
            for j in range(self.resolution):
                dist = math.sqrt((i - center)**2 + (j - center)**2)
                if abs(dist - outer_radius) < 2:
                    pattern[i, j] = intensity
        
        # Draw the triangles
        triangle_sizes = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        for size in triangle_sizes:
            radius = int(outer_radius * size)
            # Upward triangle
            self._draw_triangle(pattern, center, radius, 0 + phase_shift, intensity)
            # Downward triangle (rotated by 180 degrees)
            self._draw_triangle(pattern, center, radius, math.pi + phase_shift, intensity)
        
        # Add the central dot (bindu)
        for i in range(center-3, center+4):
            for j in range(center-3, center+4):
                if 0 <= i < self.resolution and 0 <= j < self.resolution:
                    dist = math.sqrt((i - center)**2 + (j - center)**2)
                    if dist < 3:
                        pattern[i, j] = intensity
        
        return pattern

    def _generate_quantum_field(self, intensity=1.0, phase_shift=0.0):
        """Generate a quantum field coherence pattern"""
        pattern = np.zeros((self.resolution, self.resolution), dtype=np.float64)
        center = self.resolution // 2
        
        # Create quantum interference pattern
        for i in range(self.resolution):
            for j in range(self.resolution):
                dx = (i - center) / self.resolution
                dy = (j - center) / self.resolution
                dist = math.sqrt(dx**2 + dy**2)
                
                # Create phi-based interference pattern
                angle = math.atan2(dy, dx) + phase_shift
                
                # Quantum wave function with phi-based frequency modulation
                wave1 = math.sin(2 * math.pi * dist * 10 * self.phi)
                wave2 = math.sin(2 * math.pi * angle * 5)
                wave3 = math.sin(2 * math.pi * (dist * 20 + angle * 3))
                
                # Combine waves with quantum interference
                quantum_interference = (wave1 + wave2 + wave3) / 3
                
                # Apply exponential falloff from center
                falloff = math.exp(-dist * 4)
                
                pattern[i, j] = intensity * quantum_interference * falloff
        
        # Normalize pattern
        pattern = (pattern - np.min(pattern)) / (np.max(pattern) - np.min(pattern))
        
        return pattern
    
    def _generate_golden_spiral(self, intensity=1.0, phase_shift=0.0):
        """Generate a golden spiral based on the phi ratio"""
        pattern = np.zeros((self.resolution, self.resolution), dtype=np.float64)
        center = self.resolution // 2
        max_radius = self.resolution // 2 - 10
        
        # Golden spiral is tighter than Fibonacci spiral
        b = 0.1  # Controls how tightly the spiral winds
        theta = np.linspace(0, 10 * np.pi, 1000)
        radius = max_radius * np.exp(b * theta / self.phi)
        
        for t, r in zip(theta, radius):
            angle = t + phase_shift
            x = center + int(r * math.cos(angle))
            y = center + int(r * math.sin(angle))
            
            if 0 <= x < self.resolution and 0 <= y < self.resolution:
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        xi, yi = x + dx, y + dy
                        if 0 <= xi < self.resolution and 0 <= yi < self.resolution:
                            dist = math.sqrt(dx**2 + dy**2)
                            if dist < 2.5:
                                pattern[xi, yi] = intensity * (1 - dist/2.5)
        
        return pattern
    
    def _generate_consciousness_grid(self, intensity=1.0, phase_shift=0.0):
        """Generate a consciousness alignment grid pattern"""
        pattern = np.zeros((self.resolution, self.resolution), dtype=np.float64)
        center = self.resolution // 2
        
        # Create a consciousness grid based on phi harmonics
        for i in range(self.resolution):
            for j in range(self.resolution):
                dx = (i - center) / center
                dy = (j - center) / center
                dist = math.sqrt(dx**2 + dy**2)
                
                if dist > 1.0:
                    continue
                    
                # Calculate angle from center
                angle = math.atan2(dy, dx) + phase_shift
                
                # Create phi-based grid lines
                grid_lines = 0.0
                
                # Radial lines (12 divisions)
                for k in range(12):
                    line_angle = k * (math.pi / 6)
                    angular_dist = min(abs(angle - line_angle) % math.pi, 
                                      abs(line_angle - angle) % math.pi)
                    if angular_dist < 0.05:
                        grid_lines = max(grid_lines, 0.5 * (1.0 - angular_dist / 0.05))
                
                # Concentric circles based on phi ratio
                for k in range(8):
                    # Each circle radius proportional to phi^k
                    circle_radius = 0.1 + 0.9 * (1.0 - math.pow(1.0 / self.phi, k))
                    circle_dist = abs(dist - circle_radius)
                    if circle_dist < 0.02:
                        grid_lines = max(grid_lines, 0.5 * (1.0 - circle_dist / 0.02))
                
                # Create phi-based waveform interference
                phi_wave = 0.0
                wave1 = 0.5 + 0.5 * math.cos(dist * 20 * math.pi)
                wave2 = 0.5 + 0.5 * math.cos(angle * 7 + dist * 5 * math.pi)
                wave3 = 0.5 + 0.5 * math.cos(dist * self.phi * 30 * math.pi)
                phi_wave = (wave1 + wave2 + wave3) / 3.0
                
                # Apply consciousness activation based on phi harmonics
                activation = math.pow(phi_wave * (1.0 - dist), 1.0 / self.phi)
                
                # Combine grid lines and activation fields
                pattern[i, j] = intensity * ((grid_lines * 0.7) + (activation * 0.3))
        
        # Add phi-proportioned nodal points at key intersections
        self._add_consciousness_nodes(pattern, intensity, phase_shift)
        
        # Normalize pattern
        pattern = (pattern - np.min(pattern)) / (np.max(pattern) - np.min(pattern) + 1e-10)
        
        return pattern
    
    def _add_consciousness_nodes(self, pattern, intensity=1.0, phase_shift=0.0):
        """Add consciousness activation nodes to the pattern at key phi-related points"""
        center = self.resolution // 2
        
        # Create 7 main nodes (chakra-related)
        for i in range(7):
            # Position nodes along central axis
            node_distance = 0.85 * (i / 6.0)
            node_x = center
            node_y = int(center * (1.0 - node_distance))
            
            # Node radius based on phi-relationships
            node_radius = int(self.resolution * 0.03 * (1.0 - 0.5 * (i / 6.0)))
            
            # Create glowing node effect
            for dx in range(-node_radius*2, node_radius*2 + 1):
                for dy in range(-node_radius*2, node_radius*2 + 1):
                    x, y = node_x + dx, node_y + dy
                    if 0 <= x < self.resolution and 0 <= y < self.resolution:
                        dist = math.sqrt(dx**2 + dy**2) / node_radius
                        if dist <= 2.0:
                            # Create exponential falloff glow
                            glow = math.exp(-dist * 1.5) * intensity
                            pattern[x, y] = max(pattern[x, y], glow)
        
        # Create phi-positioned auxiliary nodes in a spiral pattern
        for i in range(12):
            angle = i * (2 * math.pi / 12) + phase_shift
            dist = 0.3 + 0.3 * (i / 12.0)  # Spiral outward
            
            node_x = int(center + dist * center * math.cos(angle))
            node_y = int(center + dist * center * math.sin(angle))
            
            # Node radius varies with position
            node_radius = int(self.resolution * 0.025 * (1.0 - 0.3 * (i / 12.0)))
            
            # Create node glow
            for dx in range(-node_radius*2, node_radius*2 + 1):
                for dy in range(-node_radius*2, node_radius*2 + 1):
                    x, y = node_x + dx, node_y + dy
                    if 0 <= x < self.resolution and 0 <= y < self.resolution:
                        dist = math.sqrt(dx**2 + dy**2) / node_radius
                        if dist <= 2.0:
                            # Create exponential falloff glow
                            glow = math.exp(-dist * 1.8) * intensity * 0.8
                            pattern[x, y] = max(pattern[x, y], glow)
    
    def _generate_generic_pattern(self, pattern_type, intensity=1.0, phase_shift=0.0):
        """Generate a generic sacred geometry pattern based on the pattern type"""
        pattern = np.zeros((self.resolution, self.resolution), dtype=np.float64)
        center = self.resolution // 2
        
        # Create basic pattern based on the requested type
        for i in range(self.resolution):
            for j in range(self.resolution):
                dx = (i - center) / center
                dy = (j - center) / center
                dist = math.sqrt(dx**2 + dy**2)
                
                if dist > 1.0:
                    continue
                    
                # Calculate angle from center
                angle = math.atan2(dy, dx) + phase_shift
                
                # Generate pattern based on type
                if pattern_type == "seed_of_life":
                    # 7 overlapping circles
                    value = self._seed_of_life_pattern(dx, dy, dist, angle)
                elif pattern_type == "tree_of_life":
                    # Kabbalistic Tree of Life
                    value = self._tree_of_life_pattern(dx, dy, dist, angle)
                elif pattern_type == "vesica_piscis":
                    # Two overlapping circles
                    value = self._vesica_piscis_pattern(dx, dy, dist, angle)
                elif pattern_type == "merkaba":
                    # 3D star tetrahedron
                    value = self._merkaba_pattern(dx, dy, dist, angle)
                elif pattern_type == "torus":
                    # Toroidal energy flow
                    value = self._torus_pattern(dx, dy, dist, angle)
                elif pattern_type == "phi_grid":
                    # Phi-based grid
                    value = self._phi_grid_pattern(dx, dy, dist, angle)
                elif pattern_type == "harmonic_lattice":
                    # Harmonic frequency lattice
                    value = self._harmonic_lattice_pattern(dx, dy, dist, angle)
                else:
                    # Default sacred geometry pattern with phi-based harmonics
                    value = 0.5 + 0.5 * math.sin(angle * 5) * math.sin(dist * 10 * math.pi)
                
                pattern[i, j] = value * intensity * (1.0 - dist)
        
        # Normalize pattern
        pattern = (pattern - np.min(pattern)) / (np.max(pattern) - np.min(pattern) + 1e-10)
        
        return pattern
    
    def _draw_line(self, pattern, start, end, intensity=1.0):
        """Draw a line on the pattern between two points"""
        x0, y0 = start
        x1, y1 = end
        
        # Use Bresenham's line algorithm
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while x0 != x1 or y0 != y1:
            # Draw point if in bounds
            if 0 <= x0 < self.resolution and 0 <= y0 < self.resolution:
                pattern[x0, y0] = intensity
                
                # Draw anti-aliased edges
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        if dx == 0 and dy == 0:
                            continue
                        
                        x, y = x0 + dx, y0 + dy
                        if 0 <= x < self.resolution and 0 <= y < self.resolution:
                            dist = math.sqrt(dx**2 + dy**2)
                            pattern[x, y] = max(pattern[x, y], intensity * (1.0 - dist/2.0))
            
            # Move to next pixel
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
    
    def _draw_triangle(self, pattern, center, radius, angle_offset, intensity=1.0):
        """Draw an equilateral triangle centered at the given coordinates"""
        cx, cy = center
        
        # Calculate vertices
        vertices = []
        for i in range(3):
            angle = angle_offset + i * (2 * math.pi / 3)
            x = cx + int(radius * math.cos(angle))
            y = cy + int(radius * math.sin(angle))
            vertices.append((x, y))
        
        # Draw the triangle edges
        for i in range(3):
            self._draw_line(pattern, vertices[i], vertices[(i+1)%3], intensity)
