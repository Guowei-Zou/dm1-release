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
Pre-training MeanFlow policy
"""

import logging
import torch
log = logging.getLogger(__name__)
from agent.pretrain.train_agent import PreTrainAgent
from model.flow.meanflow import MeanFlow


class TrainMeanFlowAgent(PreTrainAgent):
    """Training agent for MeanFlow policies following ReinFlow's structure."""
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model: MeanFlow
        self.ema_model: MeanFlow
        
        self.verbose_train = False
        self.verbose_loss = False  
        self.verbose_test = True
        
        if self.test_in_mujoco:
            self.test_log_all = True
            self.only_test = False
            
            # MeanFlow follows official implementation with 5 steps by default
            self.test_denoising_steps = 5
            
            self.test_clip_intermediate_actions = True
            self.test_model_type = 'ema'
    
    def get_loss(self, batch_data):
        """Compute MeanFlow loss for training and validation."""
        # batch_data = actions, observation according to StitchedSequenceDataset
        try:
            loss = self.model.loss(*batch_data)
        except Exception as e:
            log.warning(f"MeanFlow loss computation failed: {e}. Using fallback MSE loss.")
            # Fallback to simple MSE if JVP fails
            actions, obs = batch_data
            t = torch.rand(actions.shape[0], device=actions.device)
            r = torch.zeros_like(t)
            x0 = torch.randn_like(actions)
            xt = self.model.generate_trajectory(actions, x0, t)
            u_pred = self.model.network(xt, t, r, obs)
            u_target = actions - x0
            loss = torch.nn.functional.mse_loss(u_pred, u_target)
        return loss
    
    def inference(self, cond: dict):
        """Generate samples for testing."""
        if self.test_model_type == 'ema':
            samples = self.ema_model.sample(
                cond, 
                inference_steps=self.test_denoising_steps, 
                record_intermediate=False,
                clip_intermediate_actions=self.test_clip_intermediate_actions
            )
        else:
            samples = self.model.sample(
                cond, 
                inference_steps=self.test_denoising_steps, 
                record_intermediate=False,
                clip_intermediate_actions=self.test_clip_intermediate_actions
            )
        return samples