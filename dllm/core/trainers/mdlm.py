# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from typing import Any

from dllm.core.schedulers import BaseAlphaScheduler, LinearAlphaScheduler


class MDLMTrainer(transformers.Trainer):
    """
    Base masked diffusion trainer.
    """

    def __init__(
        self,
        *args,
        scheduler: BaseAlphaScheduler | None = None,
        time_epsilon: float = 1e-3,
        loss_weight_type: str = "scheduler",  # "ones"
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.scheduler = scheduler or LinearAlphaScheduler()
        if not (0.0 < time_epsilon < 1.0):
            raise ValueError("time_epsilon must be in (0, 1)")
        self.time_epsilon = time_epsilon
        self.loss_weight_type = loss_weight_type

    def _preprocess_inputs(self, inputs):
        pass

    def _postprocess_outputs(self, outputs):
        pass

    def _compute_loss_weights(
        self,
        t: torch.Tensor,
        inputs: dict[str, Any],
        masked_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute per-token loss weights.
        """
        b, l = inputs["input_ids"].shape
        if self.loss_weight_type == "scheduler":
            loss_weights = -self.scheduler.weight(t).unsqueeze(1).repeat(1, l)
        elif self.loss_weight_type == "ones":
            loss_weights = torch.ones_like(inputs["input_ids"], dtype=torch.float32)
        else:
            raise NotImplementedError
        return loss_weights

    @torch.no_grad()
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return (loss.detach(), None, None)

        logits = getattr(outputs, "logits", outputs)
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().contiguous()

        labels = inputs.get("labels")
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().contiguous()

        return (loss.detach(), logits, labels)

    def compute_loss(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        raise NotImplementedError


from transformers import TrainerCallback
import torch


class ValidationCallback(TrainerCallback):
    def __init__(self, trainer=None):
        super().__init__()
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if hasattr(state, "is_world_process_zero") and not state.is_world_process_zero:
            return control

        trainer = self.trainer or kwargs.get("trainer")
        if trainer is None:
            print("Warning: trainer not found in callback, skipping evaluation")
            return control

        try:
            val_metrics = trainer.evaluate()
            eval_loss = val_metrics.get("eval_loss", None)
            if eval_loss is not None:
                print(f"Epoch {state.epoch:.2f} validation loss: {eval_loss:.4f}")
            else:
                print(f"Epoch {state.epoch:.2f} validation completed, but no 'eval_loss' found.")
        except Exception as e:
            print(f"Evaluation error (epoch {state.epoch:.2f}): {e}")

        return control

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from typing import Any

from dllm.core.schedulers import BaseAlphaScheduler, LinearAlphaScheduler
class StructuredMDLMTrainer(MDLMTrainer):
    """
    Structure noisy scheduling, weighted loss, mild block coupling, and step order bias for MDLM training.
    
    Training target:
    1. lower the prediction entropy of skeleton (draft) tokens, so they can be preferentially fixed by the entropy-gated decoder during inference.
    2. maintain the generation correlation within structural units (mild block coupling), 
    and introduce a weak autoregressive dependency from front to back (weaker noise and greater recovery pressure in earlier steps).
    
    Key hyperparameters:
        draft_mask_scale: float = 0.45     # draft area mask probability scale (relative to global noise level)
        draft_mask_floor: float = 0.15     # draft area mask ratio floor (minimum noise level for draft tokens)
        detail_mask_scale: float = 1.5     # detail  area mask probability scale (relative to global noise level)
        draft_loss_weight: float = 3.0     # draft token's loss weight when masked (higher to encourage better recovery)
        detail_loss_weight: float = 1.0    # detail token's loss weight when masked
        order_bias_strength: float = 0.1   # step order bias strength (how much earlier steps are favored in masking)
        coupling_factor: float = 0.3       # block coupling factor (0 = independent masking, 1 = fully coupled within blocks)
    """

    def __init__(
        self,
        *args,
        draft_mask_scale: float = 0.45,
        draft_mask_floor: float = 0.15,
        detail_mask_scale: float = 1.5,
        draft_loss_weight: float = 2.0,
        detail_loss_weight: float = 1.0,
        order_bias_strength: float = 0.1,
        coupling_factor: float = 0.3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.draft_mask_scale = draft_mask_scale
        self.draft_mask_floor = draft_mask_floor
        self.detail_mask_scale = detail_mask_scale
        self.draft_loss_weight = draft_loss_weight
        self.detail_loss_weight = detail_loss_weight
        self.order_bias_strength = order_bias_strength
        self.coupling_factor = coupling_factor



    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        self._preprocess_inputs(inputs)

        input_ids = inputs["input_ids"]           # (B, L)
        labels = inputs["labels"]                 # (B, L)
        attention_mask = inputs.get("attention_mask", None)

        b, l = input_ids.shape
        device = input_ids.device

        mask_id = getattr(self.processing_class, "mask_token_id", None)
        if mask_id is None:
            mask_id = getattr(self.tokenizer, "mask_token_id", None)
        assert mask_id is not None

        
        t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(b, device=device)
        alpha = self.scheduler(t)                      # (b,)
        p_global = (1 - alpha).unsqueeze(1).expand(b, l)  # (b, l)

        # structure-aware mask sampling
        mask = torch.zeros(b, l, dtype=torch.bool, device=device)

        
        prompt_len = inputs.get("prompt_len", None)
        prompt_mask = torch.zeros(b, l, dtype=torch.bool, device=device)
        if prompt_len is not None:
            for i, p_len in enumerate(prompt_len):
                if p_len > 0:
                    prompt_mask[i, :p_len] = True

        has_structure = ("draft_mask_pos" in inputs and "detail_mask_pos" in inputs)

        if has_structure:
            draft_pos = torch.zeros(b, l, dtype=torch.bool, device=device)
            detail_pos = torch.zeros(b, l, dtype=torch.bool, device=device)
            step_pos_list = []  #every sample: [(step_id, start, end), ...]

            for i in range(b):
                sample_steps = []
                for step_id, (s, e) in enumerate(inputs["draft_mask_pos"][i]):
                    draft_pos[i, s:e] = True
                    sample_steps.append((step_id, s, e))
                step_pos_list.append(sample_steps)
                for s, e in inputs["detail_mask_pos"][i]:
                    detail_pos[i, s:e] = True

            #basic mask ratio: draft lower，detail higher
            p_mask = p_global.clone()
            p_mask[draft_pos] = torch.clamp(
                p_global[draft_pos] * self.draft_mask_scale, self.draft_mask_floor, 1.0
            )
            p_mask[detail_pos] = torch.clamp(
                p_global[detail_pos] * self.detail_mask_scale, 0.0, 1.0
            )

            # order setting bias: earlier steps get a stronger bias (lower noise) to encourage better recovery, and create a weak autoregressive dependency
            for i in range(b):
                num_steps = len(step_pos_list[i])
                if num_steps == 0:
                    continue
                max_step_id = max(s[0] for s in step_pos_list[i])  # maximum step id for this sample
                for step_id, s, e in step_pos_list[i]:
                    # bias = (max_step_id - step_id) * self.order_bias_strength means that earlier steps (lower step_id) get a larger bias, thus lower noise, while later steps get less bias and can be noisier. 
                    bias = (max_step_id - step_id) * self.order_bias_strength
                    p_mask[i, s:e] = torch.clamp(p_mask[i, s:e] - bias, 0.05, 1.0)

            # mild block coupling:within each draft block
            for i in range(b):
                for step_id, s, e in step_pos_list[i]:
                    #for each block, generate one random number for the whole block and one for each token, then mix them according to the coupling factor.
                    block_rand = torch.rand(1, device=device).expand(e - s)
                    # personal_rand controls the independent randomness for each token, 
                    #while block_rand introduces a shared randomness that can synchronize the masking within the block. 
                    personal_rand = torch.rand(e - s, device=device)
                    # mixed_rand is a weighted combination of the independent token randomness and the shared block randomness
                    mixed_rand = (1 - self.coupling_factor) * personal_rand + \
                                 self.coupling_factor * block_rand
    
                    mask[i, s:e] = mixed_rand < p_mask[i, s:e]

                # while detail area just fully independent 
                for s, e in inputs["detail_mask_pos"][i]:
                    rand = torch.rand(e - s, device=device)
                    mask[i, s:e] = rand < p_mask[i, s:e]

        else:
            # if there is no structure information, just do global independent masking
            rand = torch.rand(b, l, device=device)
            mask = (rand < p_global) & (labels != -100)

       
        mask &= ~prompt_mask

        noised = input_ids.clone()
        noised[mask] = mask_id
        outputs = model(input_ids=noised, attention_mask=attention_mask)
        self._postprocess_outputs(outputs)
        logits = outputs.logits

        if not mask.any():
            return (logits.sum() * 0.0, outputs) if return_outputs else logits.sum() * 0.0

        token_loss = F.cross_entropy(
            logits[mask], input_ids[mask], reduction="none"
        )

        if has_structure:
            draft_masked = draft_pos[mask]
            struct_weight = torch.where(
                draft_masked, self.draft_loss_weight, self.detail_loss_weight
            )
            time_weights = self._compute_loss_weights(
                t=t, inputs=inputs, masked_indices=mask
            )
            token_loss = token_loss * struct_weight * time_weights[mask]
        else:
            loss_weights = self._compute_loss_weights(
                t=t, inputs=inputs, masked_indices=mask
            )
            token_loss = token_loss * loss_weights[mask]

        # to avoid NaN when all masked tokens are ignored, we compute the mean loss per sample and then average
 
        n_masked = mask.sum(dim=1).float()  # (B,)
        n_masked = n_masked.clamp(min=1)

        loss_per_sample = []
        start = 0
        for i in range(b):
            ni = int(n_masked[i].item())
            if ni > 0:
                loss_i = token_loss[start:start + ni].mean()
                start += ni
            else:
                loss_i = torch.tensor(0.0, device=device)
            loss_per_sample.append(loss_i)

        loss = torch.stack(loss_per_sample).mean()

        return (loss, outputs) if return_outputs else loss
