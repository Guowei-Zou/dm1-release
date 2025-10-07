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
ReFlow with Dispersive Loss integration.

This module extends the standard ReFlow with dispersive loss regularization
to encourage representation diversity in hidden spaces.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any
from model.flow.reflow import ReFlow
from model.loss.dispersive_loss import DispersiveLoss

log = logging.getLogger(__name__)


class ReFlowDispersive(nn.Module):
    """ReFlow with Dispersive Loss regularization."""
    
    def __init__(
        self,
        network,
        device,
        horizon_steps,
        action_dim,
        act_min,
        act_max,
        obs_dim,
        max_denoising_steps,
        seed,
        sample_t_type="uniform",
        # Dispersive loss parameters
        dispersive_loss_weight: float = 0.1,
        dispersive_loss_type: str = "infonce_l2", 
        dispersive_temperature: float = 0.1,
        dispersive_margin: float = 1.0,
        apply_dispersive_to_embeddings: bool = True,
        apply_dispersive_to_features: bool = False,
        **kwargs
    ):
        """
        ReFlow with dispersive loss integration.
        
        Args:
            network: Network architecture  
            dispersive_loss_weight: Weight for dispersive loss
            dispersive_loss_type: Type of dispersive loss
            dispersive_temperature: Temperature for InfoNCE
            dispersive_margin: Margin for hinge loss
            apply_dispersive_to_embeddings: Whether to apply dispersive loss to embeddings
            apply_dispersive_to_features: Whether to apply dispersive loss to intermediate features
        """
        super().__init__()
        
        # Initialize base ReFlow
        self.reflow = ReFlow(
            network=network,
            device=device,
            horizon_steps=horizon_steps,
            action_dim=action_dim,
            act_min=act_min,
            act_max=act_max,
            obs_dim=obs_dim,
            max_denoising_steps=max_denoising_steps,
            seed=seed,
            sample_t_type=sample_t_type,
            **kwargs
        )
        
        # Copy attributes for compatibility
        self.network = self.reflow.network
        self.device = self.reflow.device
        self.horizon_steps = self.reflow.horizon_steps
        self.action_dim = self.reflow.action_dim
        self.data_shape = self.reflow.data_shape
        self.act_range = self.reflow.act_range
        self.obs_dim = self.reflow.obs_dim
        self.max_denoising_steps = self.reflow.max_denoising_steps
        self.sample_t_type = self.reflow.sample_t_type
        
        # Dispersive loss configuration
        self.dispersive_loss_weight = dispersive_loss_weight
        self.apply_dispersive_to_embeddings = apply_dispersive_to_embeddings
        self.apply_dispersive_to_features = apply_dispersive_to_features
        
        # Initialize dispersive loss
        self.dispersive_loss = DispersiveLoss(
            loss_type=dispersive_loss_type,
            temperature=dispersive_temperature,
            margin=dispersive_margin,
            weight=dispersive_loss_weight,
        )
        
        log.info(f"Initialized ReFlow with Dispersive Loss: "
                f"weight={dispersive_loss_weight}, type={dispersive_loss_type}")
    
    def generate_trajectory(self, x1, x0, t):
        """Delegate to base ReFlow."""
        return self.reflow.generate_trajectory(x1, x0, t)
    
    def sample_time(self, batch_size, time_sample_type='uniform', **kwargs):
        """Delegate to base ReFlow."""
        return self.reflow.sample_time(batch_size, time_sample_type, **kwargs)
    
    def generate_target(self, x1):
        """Delegate to base ReFlow."""
        return self.reflow.generate_target(x1)
    
    def loss(self, xt, t, obs, v):
        """
        Compute ReFlow loss with dispersive regularization.
        
        Returns:
            Total loss (ReFlow loss + dispersive loss)
        """
        # Compute base ReFlow loss
        base_loss = self.reflow.loss(xt, t, obs, v)
        
        # If dispersive loss is disabled, return base loss
        if self.dispersive_loss_weight <= 0:
            return base_loss
        
        # Get batch size
        B = xt.shape[0]
        
        if B < 2:
            # Need at least 2 samples for dispersive loss
            return base_loss
        
        dispersive_losses = []
        
        try:
            # Forward pass to get embeddings from ReFlow network
            if hasattr(self.network, 'forward') and 'output_embedding' in self.network.forward.__code__.co_varnames:
                # Network supports output_embedding parameter
                try:
                    velocity, *embeddings = self.network.forward(
                        xt, t, obs, output_embedding=True
                    )
                    
                    # Apply dispersive loss to embeddings
                    if self.apply_dispersive_to_embeddings and embeddings:
                        for emb in embeddings:
                            if emb is not None and emb.requires_grad:
                                # Flatten spatial dimensions if needed
                                if emb.dim() > 2:
                                    emb_flat = emb.flatten(1)
                                else:
                                    emb_flat = emb
                                
                                emb_dispersive = self.dispersive_loss(emb_flat)
                                dispersive_losses.append(emb_dispersive)
                
                except Exception as e_inner:
                    # Fallback: try to get features from intermediate layers
                    log.debug(f"Failed to get embeddings from ReFlow network: {e_inner}")
            
            # If we have intermediate features from hooks, apply dispersive loss
            if self.apply_dispersive_to_features and hasattr(self.network, 'intermediate_features'):
                for features in self.network.intermediate_features.values():
                    if features is not None and features.requires_grad:
                        if features.dim() > 2:
                            features_flat = features.flatten(1)
                        else:
                            features_flat = features
                        feature_dispersive = self.dispersive_loss(features_flat)
                        dispersive_losses.append(feature_dispersive)
        
        except Exception as e:
            log.warning(f"Failed to compute dispersive loss for ReFlow: {e}")
        
        # Sum dispersive losses
        total_dispersive_loss = sum(dispersive_losses) if dispersive_losses else torch.tensor(0.0, device=self.device)
        
        # Combine base loss and dispersive loss
        total_loss = base_loss + total_dispersive_loss
        
        return total_loss
    
    @torch.no_grad()
    def sample(self, cond, inference_steps, record_intermediate=False, 
              clip_intermediate_actions=True, z=None):
        """Delegate to base ReFlow."""
        return self.reflow.sample(cond, inference_steps, record_intermediate, 
                                clip_intermediate_actions, z)
    
    def get_dispersive_loss_info(self) -> Dict[str, Any]:
        """Get information about dispersive loss configuration."""
        return {
            "dispersive_loss_weight": self.dispersive_loss_weight,
            "dispersive_loss_type": self.dispersive_loss.loss_type,
            "dispersive_temperature": self.dispersive_loss.temperature,
            "dispersive_margin": self.dispersive_loss.margin,
            "apply_to_embeddings": self.apply_dispersive_to_embeddings,
            "apply_to_features": self.apply_dispersive_to_features,
        }