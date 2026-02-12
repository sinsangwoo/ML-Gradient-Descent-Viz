"""
Loss Landscape Analysis and Visualization

Provides tools to:
1. Generate 2D/3D loss landscapes
2. Identify critical points (local minima, saddle points)
3. Visualize optimization trajectories
4. Analyze Hessian eigenvalues

References:
- Li et al. (2018). "Visualizing the Loss Landscape of Neural Nets"
- Dauphin et al. (2014). "Identifying and attacking saddle points"
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Callable, Tuple, List, Dict, Optional
import warnings


class LossLandscapeAnalyzer:
    """
    Analyze and visualize loss landscapes for optimization problems.
    """
    
    def __init__(self, loss_fn: Callable, gradient_fn: Optional[Callable] = None):
        """
        Parameters:
        -----------
        loss_fn : callable
            Function that takes parameters and returns loss
            Signature: loss = loss_fn(params)
        gradient_fn : callable, optional
            Function that computes gradient
            Signature: grad = gradient_fn(params)
        """
        self.loss_fn = loss_fn
        self.gradient_fn = gradient_fn
    
    def generate_2d_landscape(self, center: np.ndarray, 
                             direction1: np.ndarray, direction2: np.ndarray,
                             alpha_range: Tuple[float, float] = (-1, 1),
                             beta_range: Tuple[float, float] = (-1, 1),
                             resolution: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate 2D loss landscape along two directions.
        
        params = center + alpha * direction1 + beta * direction2
        
        Parameters:
        -----------
        center : ndarray
            Central point in parameter space
        direction1, direction2 : ndarray
            Directions to explore (should be normalized)
        alpha_range, beta_range : tuple
            Range of coefficients to explore
        resolution : int
            Grid resolution
            
        Returns:
        --------
        alphas, betas : ndarray
            Grid coordinates
        losses : ndarray
            Loss values at each grid point
        """
        # Create grid
        alphas = np.linspace(alpha_range[0], alpha_range[1], resolution)
        betas = np.linspace(beta_range[0], beta_range[1], resolution)
        Alpha, Beta = np.meshgrid(alphas, betas)
        
        # Compute losses
        losses = np.zeros_like(Alpha)
        
        for i in range(resolution):
            for j in range(resolution):
                params = center + Alpha[i, j] * direction1 + Beta[i, j] * direction2
                losses[i, j] = self.loss_fn(params)
        
        return Alpha, Beta, losses
    
    def visualize_landscape_2d(self, Alpha: np.ndarray, Beta: np.ndarray, 
                               losses: np.ndarray, trajectory: Optional[List] = None,
                               title: str = "Loss Landscape",
                               save_path: Optional[str] = None):
        """
        Visualize 2D loss landscape as contour plot.
        
        Parameters:
        -----------
        Alpha, Beta : ndarray
            Grid coordinates from generate_2d_landscape
        losses : ndarray
            Loss values
        trajectory : list of ndarray, optional
            Optimization trajectory to overlay
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Contour plot
        contour = ax.contour(Alpha, Beta, losses, levels=30, cmap='viridis', alpha=0.6)
        contourf = ax.contourf(Alpha, Beta, losses, levels=30, cmap='viridis', alpha=0.4)
        
        # Color bar
        cbar = plt.colorbar(contourf, ax=ax)
        cbar.set_label('Loss', rotation=270, labelpad=20, fontsize=12)
        
        # Add contour labels
        ax.clabel(contour, inline=True, fontsize=8)
        
        # Overlay trajectory if provided
        if trajectory is not None and len(trajectory) > 0:
            traj_alpha = [p[0] for p in trajectory]
            traj_beta = [p[1] for p in trajectory]
            ax.plot(traj_alpha, traj_beta, 'r.-', linewidth=2, 
                   markersize=8, label='Optimization Path', alpha=0.8)
            ax.plot(traj_alpha[0], traj_beta[0], 'go', markersize=12, 
                   label='Start', zorder=10)
            ax.plot(traj_alpha[-1], traj_beta[-1], 'r*', markersize=15, 
                   label='End', zorder=10)
            ax.legend(fontsize=10)
        
        ax.set_xlabel(r'$\alpha$ (Direction 1)', fontsize=12)
        ax.set_ylabel(r'$\beta$ (Direction 2)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved landscape to '{save_path}'")
        
        plt.show()
    
    def visualize_landscape_3d(self, Alpha: np.ndarray, Beta: np.ndarray,
                               losses: np.ndarray, title: str = "3D Loss Landscape",
                               save_path: Optional[str] = None):
        """
        Visualize loss landscape as 3D surface.
        """
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Surface plot
        surf = ax.plot_surface(Alpha, Beta, losses, cmap='viridis',
                              alpha=0.8, edgecolor='none')
        
        # Color bar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label('Loss', rotation=270, labelpad=20, fontsize=12)
        
        ax.set_xlabel(r'$\alpha$ (Direction 1)', fontsize=11)
        ax.set_ylabel(r'$\beta$ (Direction 2)', fontsize=11)
        ax.set_zlabel('Loss', fontsize=11)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Viewing angle
        ax.view_init(elev=30, azim=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved 3D landscape to '{save_path}'")
        
        plt.show()
    
    def compute_hessian(self, params: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
        """
        Compute Hessian matrix using finite differences.
        
        H[i,j] = ∂²L / ∂θ_i ∂θ_j
        
        Parameters:
        -----------
        params : ndarray
            Parameters at which to compute Hessian
        epsilon : float
            Step size for finite differences
            
        Returns:
        --------
        H : ndarray, shape (d, d)
            Hessian matrix
        """
        d = len(params)
        H = np.zeros((d, d))
        
        for i in range(d):
            for j in range(d):
                # Compute ∂²L / ∂θ_i ∂θ_j using finite differences
                
                # f(x + ei + ej)
                params_pp = params.copy()
                params_pp[i] += epsilon
                params_pp[j] += epsilon
                f_pp = self.loss_fn(params_pp)
                
                # f(x + ei - ej)
                params_pm = params.copy()
                params_pm[i] += epsilon
                params_pm[j] -= epsilon
                f_pm = self.loss_fn(params_pm)
                
                # f(x - ei + ej)
                params_mp = params.copy()
                params_mp[i] -= epsilon
                params_mp[j] += epsilon
                f_mp = self.loss_fn(params_mp)
                
                # f(x - ei - ej)
                params_mm = params.copy()
                params_mm[i] -= epsilon
                params_mm[j] -= epsilon
                f_mm = self.loss_fn(params_mm)
                
                # Central difference formula
                H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * epsilon**2)
        
        return H
    
    def classify_critical_point(self, params: np.ndarray, 
                               epsilon: float = 1e-5) -> Dict:
        """
        Classify critical point using Hessian eigenvalues.
        
        Classification:
        - All eigenvalues > 0: Local minimum
        - All eigenvalues < 0: Local maximum
        - Mixed signs: Saddle point
        
        Parameters:
        -----------
        params : ndarray
            Point to classify
        epsilon : float
            Step size for Hessian computation
            
        Returns:
        --------
        result : dict
            Contains: type, eigenvalues, gradient_norm
        """
        # Compute Hessian
        H = self.compute_hessian(params, epsilon)
        
        # Eigenvalue decomposition
        eigenvalues = np.linalg.eigvals(H)
        eigenvalues_real = np.real(eigenvalues)
        
        # Compute gradient norm if gradient_fn available
        if self.gradient_fn is not None:
            grad = self.gradient_fn(params)
            grad_norm = np.linalg.norm(grad)
        else:
            grad_norm = None
        
        # Classify
        positive = np.sum(eigenvalues_real > 1e-6)
        negative = np.sum(eigenvalues_real < -1e-6)
        
        if positive == len(eigenvalues):
            point_type = "Local Minimum"
        elif negative == len(eigenvalues):
            point_type = "Local Maximum"
        else:
            point_type = f"Saddle Point ({negative} negative, {positive} positive)"
        
        return {
            'type': point_type,
            'eigenvalues': eigenvalues_real,
            'min_eigenvalue': np.min(eigenvalues_real),
            'max_eigenvalue': np.max(eigenvalues_real),
            'gradient_norm': grad_norm,
            'hessian': H
        }


if __name__ == "__main__":
    # Test with Rosenbrock function (classic non-convex test)
    print("Testing Loss Landscape Analyzer\n")
    
    def rosenbrock(params):
        """Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2"""
        x, y = params[0], params[1]
        return (1 - x)**2 + 100 * (y - x**2)**2
    
    def rosenbrock_grad(params):
        """Gradient of Rosenbrock."""
        x, y = params[0], params[1]
        dx = -2*(1-x) - 400*x*(y - x**2)
        dy = 200*(y - x**2)
        return np.array([dx, dy])
    
    # Create analyzer
    analyzer = LossLandscapeAnalyzer(rosenbrock, rosenbrock_grad)
    
    # Generate landscape around a point
    center = np.array([0.0, 0.0])
    direction1 = np.array([1.0, 0.0])  # x-direction
    direction2 = np.array([0.0, 1.0])  # y-direction
    
    print("Generating 2D landscape...")
    Alpha, Beta, losses = analyzer.generate_2d_landscape(
        center, direction1, direction2,
        alpha_range=(-2, 2), beta_range=(-1, 3),
        resolution=100
    )
    
    print("\nVisualizing landscape...")
    analyzer.visualize_landscape_2d(Alpha, Beta, losses, 
                                    title="Rosenbrock Function Landscape")
    
    # Classify critical point at origin
    print("\nAnalyzing critical point at (0, 0)...")
    result = analyzer.classify_critical_point(center)
    print(f"Type: {result['type']}")
    print(f"Eigenvalues: {result['eigenvalues']}")
    print(f"Gradient norm: {result['gradient_norm']:.6f}")