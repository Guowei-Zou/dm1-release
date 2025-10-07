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
ShortCut Flow with Dispersive Loss integration.

This module extends the standard ShortCut Flow with dispersive loss regularization
to encourage representation diversity in hidden spaces.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Tuple, Optional
from model.flow.shortcutflow import ShortCutFlow
from model.loss.dispersive_loss import DispersiveLoss

log = logging.getLogger(__name__)


class ShortCutFlowDispersive(ShortCutFlow):
    """ShortCut Flow with Dispersive Loss regularization."""
    
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
        self_consistency_k=0.25,
        delta=1e-5,
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
        Args:
            network: Network architecture
            dispersive_loss_weight: Weight coefficient for dispersive loss
            dispersive_loss_type: Type of dispersive loss ("infonce_l2", "infonce_cosine", "hinge", "covariance")
            dispersive_temperature: Temperature parameter for InfoNCE variants
            dispersive_margin: Margin parameter for hinge loss
            apply_dispersive_to_embeddings: Apply dispersive loss to condition embeddings
            apply_dispersive_to_features: Apply dispersive loss to intermediate features
        """
        super().__init__(
            network=network,
            device=device,
            horizon_steps=horizon_steps,
            action_dim=action_dim,
            act_min=act_min,
            act_max=act_max,
            obs_dim=obs_dim,
            max_denoising_steps=max_denoising_steps,
            seed=seed,
            self_consistency_k=self_consistency_k,
            delta=delta,
            sample_t_type=sample_t_type,
            **kwargs
        )
        
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
        
        log.info(f"Initialized ShortCut Flow with Dispersive Loss: "
                f"weight={dispersive_loss_weight}, type={dispersive_loss_type}")
    
    def loss(self, action, cond, **kwargs):
        """
        Compute ShortCut Flow loss with dispersive regularization.
        
        Args:
            action: (B, Ta, Da) action trajectories
            cond: Condition dictionary
            
        Returns:
            Total loss (ShortCut loss + dispersive loss)
        """
        # Compute base ShortCut Flow loss
        base_loss = super().loss(action, cond)
        
        # If dispersive loss is disabled, return base loss
        if self.dispersive_loss_weight <= 0:
            return base_loss
        
        # Get batch size
        B = action.shape[0]
        
        if B < 2:
            # Need at least 2 samples for dispersive loss
            return base_loss
        
        dispersive_losses = []
        
        try:
            # For ShortCut Flow, we sample time and step size similar to its loss method
            k_num = int(B * self.self_consistency_k)
            
            # Sample times for self-consistency part
            t_sc = torch.rand(k_num, device=self.device)
            d_sc = torch.rand(k_num, device=self.device) * 0.5  # Random step sizes
            
            # Sample times for flow-matching part  
            t_fm = torch.rand(B - k_num, device=self.device)
            d_fm = torch.zeros(B - k_num, device=self.device)
            
            # Combine
            t = torch.cat([t_sc, t_fm], dim=0)
            dt = torch.cat([d_sc, d_fm], dim=0)
            
            # Forward pass to get embeddings
            if hasattr(self.network, 'forward') and 'output_embedding' in self.network.forward.__code__.co_varnames:
                # Network supports output_embedding parameter
                try:
                    velocity, td_emb, cond_emb = self.network.forward(
                        action, t, dt, cond, output_embedding=True
                    )
                    
                    # Apply dispersive loss to embeddings
                    if self.apply_dispersive_to_embeddings:
                        # Apply to time-step embeddings
                        if td_emb is not None:
                            td_emb_flat = td_emb.view(B, -1)
                            td_dispersive = self.dispersive_loss(td_emb_flat)
                            dispersive_losses.append(td_dispersive)
                        
                        # Apply to condition embeddings
                        if cond_emb is not None:
                            cond_emb_flat = cond_emb.view(B, -1)
                            cond_dispersive = self.dispersive_loss(cond_emb_flat)
                            dispersive_losses.append(cond_dispersive)
                            
                except (TypeError, AttributeError):
                    # Network doesn't support output_embedding
                    pass
                        
        except Exception as e:
            log.warning(f"Failed to compute dispersive loss: {e}")
            
        # Sum dispersive losses
        total_dispersive_loss = sum(dispersive_losses) if dispersive_losses else torch.tensor(0.0, device=self.device)
        
        # Combine base loss and dispersive loss
        total_loss = base_loss + total_dispersive_loss
        
        return total_loss
    
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


class MeanFlowDispersive(nn.Module):
    """MeanFlow with Dispersive Loss regularization."""
    
    def __init__(
        self,
        network,
        device,
        horizon_steps,
        action_dim,
        act_min,
        act_max,
        obs_dim,
        max_denoising_steps=1,
        seed=42,
        flow_ratio=0.5,
        gamma=0.5,
        c=1e-3,
        sample_t_type="uniform",
        # Dispersive loss parameters
        dispersive_loss_weight: float = 0.1,
        dispersive_loss_type: str = "infonce_l2",
        dispersive_temperature: float = 0.1,
        dispersive_margin: float = 1.0,
        apply_dispersive_to_embeddings: bool = True,
        **kwargs
    ):
        """
        MeanFlow with dispersive loss integration.
        
        Args:
            network: MeanFlow network
            dispersive_loss_weight: Weight for dispersive loss
            dispersive_loss_type: Type of dispersive loss
            dispersive_temperature: Temperature for InfoNCE
            dispersive_margin: Margin for hinge loss
            apply_dispersive_to_embeddings: Whether to apply dispersive loss to embeddings
        """
        super().__init__()
        
        # Import MeanFlow here to avoid circular imports
        from model.flow.meanflow import MeanFlow
        
        # Initialize base MeanFlow
        self.base_meanflow = MeanFlow(
            network=network,
            device=device,
            horizon_steps=horizon_steps,
            action_dim=action_dim,
            act_min=act_min,
            act_max=act_max,
            obs_dim=obs_dim,
            max_denoising_steps=max_denoising_steps,
            seed=seed,
            flow_ratio=flow_ratio,
            gamma=gamma,
            c=c,
            sample_t_type=sample_t_type,
            **kwargs
        )
        
        # Copy attributes from base model
        self.network = self.base_meanflow.network
        self.device = self.base_meanflow.device
        self.horizon_steps = self.base_meanflow.horizon_steps
        self.action_dim = self.base_meanflow.action_dim
        self.act_range = self.base_meanflow.act_range
        self.obs_dim = self.base_meanflow.obs_dim
        self.flow_ratio = self.base_meanflow.flow_ratio
        self.gamma = self.base_meanflow.gamma
        self.c = self.base_meanflow.c
        
        # Dispersive loss configuration
        self.dispersive_loss_weight = dispersive_loss_weight
        self.apply_dispersive_to_embeddings = apply_dispersive_to_embeddings
        
        # Initialize dispersive loss
        self.dispersive_loss = DispersiveLoss(
            loss_type=dispersive_loss_type,
            temperature=dispersive_temperature,
            margin=dispersive_margin,
            weight=dispersive_loss_weight,
        )
        
        log.info(f"Initialized MeanFlow with Dispersive Loss: "
                f"weight={dispersive_loss_weight}, type={dispersive_loss_type}")
    
    def forward(self, *args, **kwargs):
        """Forward pass through base MeanFlow."""
        return self.base_meanflow.forward(*args, **kwargs)
    
    def sample(self, *args, **kwargs):
        """Sample from base MeanFlow."""
        return self.base_meanflow.sample(*args, **kwargs)
    
    def loss(self, action, cond, **kwargs):
        """
        Compute MeanFlow loss with dispersive regularization.
        """
        # Compute base MeanFlow loss
        base_loss = self.base_meanflow.loss(action, cond, **kwargs)
        
        # If dispersive loss is disabled, return base loss
        if self.dispersive_loss_weight <= 0:
            return base_loss
        
        # Get batch size
        B = action.shape[0]
        
        if B < 2:
            # Need at least 2 samples for dispersive loss
            return base_loss
        
        dispersive_losses = []
        
        try:
            # Get embeddings from network if available
            with torch.no_grad():
                if hasattr(self.network, 'forward') and 'output_embedding' in self.network.forward.__code__.co_varnames:
                    # Sample parameters for forward pass
                    t = torch.rand(B, device=self.device)
                    r = torch.rand(B, device=self.device)
                    
                    try:
                        velocity, time_emb, r_emb, cond_emb = self.network.forward(
                            action, t, r, cond, output_embedding=True
                        )
                        
                        # Apply dispersive loss to embeddings
                        if self.apply_dispersive_to_embeddings:
                            # Apply to time embeddings
                            if time_emb is not None and time_emb.requires_grad:
                                time_emb_flat = time_emb.view(B, -1)
                                time_dispersive = self.dispersive_loss(time_emb_flat)
                                dispersive_losses.append(time_dispersive)
                            
                            # Apply to r embeddings  
                            if r_emb is not None and r_emb.requires_grad:
                                r_emb_flat = r_emb.view(B, -1)
                                r_dispersive = self.dispersive_loss(r_emb_flat)
                                dispersive_losses.append(r_dispersive)
                            
                            # Apply to condition embeddings
                            if cond_emb is not None and cond_emb.requires_grad:
                                cond_emb_flat = cond_emb.view(B, -1)
                                cond_dispersive = self.dispersive_loss(cond_emb_flat)
                                dispersive_losses.append(cond_dispersive)
                                
                    except TypeError:
                        # Network doesn't support output_embedding
                        pass
                        
        except Exception as e:
            log.warning(f"Failed to compute dispersive loss for MeanFlow: {e}")
            
        # Sum dispersive losses
        total_dispersive_loss = sum(dispersive_losses) if dispersive_losses else torch.tensor(0.0, device=self.device)
        
        # Combine base loss and dispersive loss
        total_loss = base_loss + total_dispersive_loss
        
        return total_loss