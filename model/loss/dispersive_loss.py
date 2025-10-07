# MIT License

# Copyright (c) 2025 ReinFlow Authors

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Dispersive Loss implementation based on "Diffuse and Disperse: Image Generation with Representation Regularization".

Dispersive loss is a plug-and-play regularizer that encourages internal representations 
to disperse in hidden space, similar to contrastive learning but without requiring positive pairs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Literal

log = logging.getLogger(__name__)


class DispersiveLoss(nn.Module):
    """
    Dispersive Loss implementation with multiple variants:
    - InfoNCE with L2 distance
    - InfoNCE with cosine distance  
    - Hinge loss
    - Covariance loss
    """
    
    def __init__(
        self,
        loss_type: Literal["infonce_l2", "infonce_cosine", "hinge", "covariance"] = "infonce_l2",
        temperature: float = 0.5,  # Optimal temperature from "Diffuse and Disperse" paper
        margin: float = 1.0,
        weight: float = 1.0,
        eps: float = 1e-8,
    ):
        """
        Args:
            loss_type: Type of dispersive loss to use
            temperature: Temperature parameter for InfoNCE variants
            margin: Margin parameter for hinge loss
            weight: Weight coefficient for the dispersive loss
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.loss_type = loss_type
        self.temperature = temperature
        self.margin = margin
        self.weight = weight
        self.eps = eps
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute dispersive loss on internal representations.
        
        Args:
            features: Hidden representations (B, D) where B is batch size, D is feature dimension
            
        Returns:
            Dispersive loss scalar
        """
        B, D = features.shape
        
        if B < 2:
            # Need at least 2 samples for dispersive loss
            return torch.tensor(0.0, device=features.device, requires_grad=True)
            
        if self.loss_type == "infonce_l2":
            return self._infonce_l2_loss(features)
        elif self.loss_type == "infonce_cosine":
            return self._infonce_cosine_loss(features)
        elif self.loss_type == "hinge":
            return self._hinge_loss(features)
        elif self.loss_type == "covariance":
            return self._covariance_loss(features)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
    
    def _infonce_l2_loss(self, features: torch.Tensor) -> torch.Tensor:
        """InfoNCE loss with L2 distance."""
        B = features.shape[0]
        
        # Compute pairwise L2 distances
        features_norm = features.unsqueeze(1)  # (B, 1, D)
        features_norm_t = features.unsqueeze(0)  # (1, B, D)
        distances = torch.norm(features_norm - features_norm_t, dim=2)  # (B, B)
        
        # Convert distances to similarities (negative distances)
        similarities = -distances / self.temperature
        
        # Mask out diagonal (self-similarities)
        mask = torch.eye(B, device=features.device, dtype=torch.bool)
        similarities = similarities.masked_fill(mask, -1e9)  # Use large negative instead of -inf
        
        # InfoNCE: we want to minimize similarities (encourage dispersion)
        # Clamp to avoid numerical issues
        similarities = torch.clamp(similarities, min=-10, max=10)
        loss = -torch.mean(torch.logsumexp(-similarities, dim=1))  # Fixed: added negative sign
        
        return self.weight * loss
    
    def _infonce_cosine_loss(self, features: torch.Tensor) -> torch.Tensor:
        """InfoNCE loss with cosine similarity."""
        B = features.shape[0]
        
        # L2 normalize features
        features_norm = F.normalize(features, p=2, dim=1, eps=self.eps)
        
        # Compute cosine similarities
        similarities = torch.mm(features_norm, features_norm.t()) / self.temperature  # (B, B)
        
        # Mask out diagonal
        mask = torch.eye(B, device=features.device, dtype=torch.bool)
        similarities = similarities.masked_fill(mask, -1e9)  # Use large negative instead of -inf
        
        # InfoNCE: we want to minimize similarities (encourage dispersion)
        # Clamp to avoid numerical issues
        similarities = torch.clamp(similarities, min=-10, max=10)
        loss = -torch.mean(torch.logsumexp(-similarities, dim=1))
        
        return self.weight * loss
    
    def _hinge_loss(self, features: torch.Tensor) -> torch.Tensor:
        """Hinge loss variant for dispersive regularization."""
        B = features.shape[0]
        
        # Compute pairwise L2 distances
        features_norm = features.unsqueeze(1)  # (B, 1, D)
        features_norm_t = features.unsqueeze(0)  # (1, B, D)
        distances = torch.norm(features_norm - features_norm_t, dim=2)  # (B, B)
        
        # Mask out diagonal
        mask = torch.eye(B, device=features.device, dtype=torch.bool)
        distances = distances.masked_fill(mask, float('inf'))
        
        # Hinge loss: encourage distances to be at least margin
        # loss = max(0, margin - distance)
        hinge_losses = F.relu(self.margin - distances)
        
        # Average over all pairs (excluding diagonal)
        loss = torch.mean(hinge_losses[~mask])
        
        return self.weight * loss
    
    def _covariance_loss(self, features: torch.Tensor) -> torch.Tensor:
        """Covariance-based dispersive loss."""
        B, D = features.shape
        
        # Center the features
        features_centered = features - torch.mean(features, dim=0, keepdim=True)
        
        # Compute covariance matrix
        cov_matrix = torch.mm(features_centered.t(), features_centered) / (B - 1)
        
        # Encourage off-diagonal elements to be zero (decorrelation)
        # and diagonal elements to be large (high variance)
        off_diagonal_mask = ~torch.eye(D, device=features.device, dtype=torch.bool)
        
        # Loss = mean of squared off-diagonal elements + penalty for small diagonal elements
        # We want: small off-diagonal (decorrelation) and large diagonal (high variance)
        off_diagonal_loss = torch.mean(cov_matrix[off_diagonal_mask] ** 2)
        diagonal_loss = torch.mean(torch.relu(1.0 - torch.diag(cov_matrix)))  # penalize diagonal < 1
        
        loss = off_diagonal_loss + diagonal_loss
        
        return self.weight * loss


class DispersiveLossWrapper(nn.Module):
    """
    Wrapper to apply dispersive loss to intermediate representations of a model.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        dispersive_loss: DispersiveLoss,
        hook_layer_names: list = None,
        apply_to_final_features: bool = True,
    ):
        """
        Args:
            base_model: The base model to wrap
            dispersive_loss: DispersiveLoss instance
            hook_layer_names: List of layer names to apply dispersive loss to
            apply_to_final_features: Whether to apply dispersive loss to final features
        """
        super().__init__()
        self.base_model = base_model
        self.dispersive_loss = dispersive_loss
        self.hook_layer_names = hook_layer_names or []
        self.apply_to_final_features = apply_to_final_features
        
        self.intermediate_features = {}
        self.hooks = []
        
        # Register hooks if specified
        if self.hook_layer_names:
            self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture intermediate features."""
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    # Flatten spatial dimensions if needed
                    if output.dim() > 2:
                        output_flat = output.flatten(1)
                    else:
                        output_flat = output
                    self.intermediate_features[name] = output_flat
            return hook
        
        for name, module in self.base_model.named_modules():
            if name in self.hook_layer_names:
                handle = module.register_forward_hook(hook_fn(name))
                self.hooks.append(handle)
    
    def forward(self, *args, **kwargs):
        """Forward pass with dispersive loss computation."""
        # Clear previous intermediate features
        self.intermediate_features.clear()
        
        # Forward pass through base model
        if hasattr(self.base_model, 'forward'):
            if 'output_embedding' in kwargs:
                # Handle models that can output embeddings
                original_output_embedding = kwargs.get('output_embedding', False)
                kwargs['output_embedding'] = True
                result = self.base_model(*args, **kwargs)
                
                if isinstance(result, tuple) and len(result) > 1:
                    output, *embeddings = result
                    final_features = embeddings[-1] if embeddings else None
                else:
                    output = result
                    final_features = None
                
                # Restore original output_embedding setting
                if not original_output_embedding and isinstance(result, tuple):
                    output = result[0]
            else:
                output = self.base_model(*args, **kwargs)
                final_features = None
        else:
            output = self.base_model(*args, **kwargs)
            final_features = None
        
        # Compute dispersive losses
        dispersive_losses = []
        
        # Apply to intermediate features
        for name, features in self.intermediate_features.items():
            if features is not None and features.requires_grad:
                disp_loss = self.dispersive_loss(features)
                dispersive_losses.append(disp_loss)
        
        # Apply to final features if requested
        if self.apply_to_final_features and final_features is not None:
            if final_features.dim() > 2:
                final_features = final_features.flatten(1)
            disp_loss = self.dispersive_loss(final_features)
            dispersive_losses.append(disp_loss)
        
        # Sum all dispersive losses
        total_dispersive_loss = sum(dispersive_losses) if dispersive_losses else torch.tensor(0.0, device=output.device)
        
        return output, total_dispersive_loss
    
    def __del__(self):
        """Clean up hooks."""
        for hook in self.hooks:
            hook.remove()


def test_dispersive_loss():
    """Test function for dispersive loss variants."""
    print("Testing Dispersive Loss implementations...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 8
    feature_dim = 128
    
    # Generate random features
    features = torch.randn(batch_size, feature_dim, device=device, requires_grad=True)
    
    # Test different loss types
    loss_types = ["infonce_l2", "infonce_cosine", "hinge", "covariance"]
    
    for loss_type in loss_types:
        print(f"\nTesting {loss_type}:")
        dispersive_loss = DispersiveLoss(loss_type=loss_type, weight=1.0).to(device)
        
        loss_value = dispersive_loss(features)
        print(f"  Loss value: {loss_value.item():.6f}")
        
        # Test gradient computation
        loss_value.backward(retain_graph=True)
        if features.grad is not None:
            print(f"  Gradient norm: {features.grad.norm().item():.6f}")
            features.grad.zero_()
        else:
            print("  No gradients computed")
    
    print("\nDispersive Loss tests completed!")


if __name__ == "__main__":
    test_dispersive_loss()