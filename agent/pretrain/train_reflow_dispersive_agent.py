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
Pre-training ReFlow policy with Dispersive Loss regularization
"""
import logging
log = logging.getLogger(__name__)
from agent.pretrain.train_agent import PreTrainAgent
from model.flow.reflow_dispersive import ReFlowDispersive


class TrainReFlowDispersiveAgent(PreTrainAgent):
    """Training agent for ReFlow with Dispersive Loss."""
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model: ReFlowDispersive
        self.ema_model: ReFlowDispersive
        
        self.verbose_train = False  # True #False #True #False #True #False
        self.verbose_loss = True   # False #False #True #False #True #False # True
        self.verbose_test = False  # True #False
        
        if self.test_in_mujoco:
            self.test_log_all = True
            self.only_test = False                      # when toggled, test and then exit code.
            
            self.test_denoising_steps = 4               # self.model.max_denoising_steps!!!!!!!!!!!!
            
            self.test_clip_intermediate_actions = True  # True    # this will affect performance. !!!!!!!!!!!!
            
            self.test_model_type = 'ema'                # minor difference!!!!!!!!!!!!
        
        # Log dispersive loss configuration
        dispersive_info = self.model.get_dispersive_loss_info()
        log.info(f"ReFlow Dispersive Loss Configuration: {dispersive_info}")

    def get_model_loss(self, action, cond):
        """
        Compute ReFlow loss with dispersive regularization.
        
        Args:
            action: (B, Ta, Da) action trajectories
            cond: Condition dictionary
            
        Returns:
            Loss tensor with dispersive regularization
        """
        (xt, t), v = self.model.generate_target(action)
        
        # The ReFlowDispersive loss method handles both base loss and dispersive loss
        loss = self.model.loss(xt, t, cond, v)
        
        return loss

    def get_ema_model_loss(self, action, cond):
        """
        Compute EMA ReFlow loss with dispersive regularization.
        
        Args:
            action: (B, Ta, Da) action trajectories
            cond: Condition dictionary
            
        Returns:
            Loss tensor with dispersive regularization
        """
        (xt, t), v = self.ema_model.generate_target(action)
        
        # The ReFlowDispersive loss method handles both base loss and dispersive loss
        loss = self.ema_model.loss(xt, t, cond, v)
        
        return loss
    
    def log_additional_train_info(self, epoch, loss_dict):
        """Log additional training information including dispersive loss details."""
        super().log_additional_train_info(epoch, loss_dict)
        
        # Log dispersive loss configuration periodically
        if epoch % 100 == 0:
            dispersive_info = self.model.get_dispersive_loss_info()
            log.info(f"Epoch {epoch} - Dispersive Loss Config: {dispersive_info}")