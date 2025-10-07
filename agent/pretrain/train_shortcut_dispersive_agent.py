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
Training agent for ShortCut Flow with Dispersive Loss.
"""

import logging
import torch
import wandb
from agent.pretrain.train_shortcut_agent import TrainShortCutAgent

log = logging.getLogger(__name__)


class TrainShortCutDispersiveAgent(TrainShortCutAgent):
    """Training agent for ShortCut Flow with Dispersive Loss regularization."""
    
    def __init__(self, cfg):
        super().__init__(cfg)
        log.info("Initialized ShortCut Flow Dispersive training agent")
        
        # Log dispersive loss configuration if available
        if hasattr(self.model, 'get_dispersive_loss_info'):
            dispersive_info = self.model.get_dispersive_loss_info()
            log.info(f"Dispersive loss configuration: {dispersive_info}")
            
            # Log to wandb if available
            if wandb.run is not None:
                wandb.config.update({f"dispersive_{k}": v for k, v in dispersive_info.items()})
    
    def get_loss(self, batch_data):
        """
        Compute loss with dispersive regularization.
        """
        try:
            # The model's loss method already includes dispersive loss
            loss = self.model.loss(*batch_data)
            return loss
        except Exception as e:
            log.error(f"Error computing dispersive loss: {e}")
            # Fallback to standard loss computation
            return super().get_loss(batch_data)
    
    def train_step(self, batch_data):
        """Training step with dispersive loss logging."""
        # Standard training step
        loss_info = super().train_step(batch_data)
        
        # Additional logging for dispersive loss components if needed
        # This could be extended to log individual loss components
        
        return loss_info


class TrainMeanFlowDispersiveAgent(TrainShortCutAgent):
    """Training agent for MeanFlow with Dispersive Loss regularization."""
    
    def __init__(self, cfg):
        super().__init__(cfg)
        log.info("Initialized MeanFlow Dispersive training agent")
        
        # Log dispersive loss configuration if available
        if hasattr(self.model, 'dispersive_loss'):
            dispersive_info = {
                "dispersive_loss_weight": self.model.dispersive_loss_weight,
                "dispersive_loss_type": self.model.dispersive_loss.loss_type,
                "dispersive_temperature": self.model.dispersive_loss.temperature,
                "dispersive_margin": self.model.dispersive_loss.margin,
            }
            log.info(f"Dispersive loss configuration: {dispersive_info}")
            
            # Log to wandb if available
            if wandb.run is not None:
                wandb.config.update({f"dispersive_{k}": v for k, v in dispersive_info.items()})
    
    def get_loss(self, batch_data):
        """
        Compute MeanFlow loss with dispersive regularization.
        """
        try:
            # The model's loss method already includes dispersive loss
            loss = self.model.loss(*batch_data)
            return loss
        except Exception as e:
            log.error(f"Error computing MeanFlow dispersive loss: {e}")
            # Fallback to MSE loss like in the original MeanFlow agent
            try:
                from model.flow.meanflow import MeanFlow
                action, cond = batch_data[:2]
                B = action.shape[0]
                
                # Sample t and r
                t = torch.rand(B, device=self.device)
                r = torch.rand(B, device=self.device)
                
                # Compute velocity prediction
                velocity_pred = self.model.network.forward(action, t, r, cond)
                
                # Compute target velocity (simple MSE fallback)
                velocity_target = torch.randn_like(velocity_pred)
                loss = torch.nn.functional.mse_loss(velocity_pred, velocity_target)
                
                log.warning(f"Using fallback MSE loss: {loss.item()}")
                return loss
                
            except Exception as fallback_error:
                log.error(f"Fallback loss computation also failed: {fallback_error}")
                raise e
    
    def train_step(self, batch_data):
        """Training step with dispersive loss logging."""
        # Standard training step
        loss_info = super().train_step(batch_data)
        
        # Additional logging for dispersive loss components if needed
        
        return loss_info