import json
import math
from dataclasses import dataclass, field
import logging
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Callable, Dict, List, Optional, Tuple, Union
import inspect
from torch import nn, Tensor
from nemo.utils.import_utils import safe_import
_, HAVE_TE = safe_import("transformer_engine")
import torch
from nemo.collections.llm.gpt.model.base import GPTConfig, GPTModel, torch_dtype_from_mcore_config, transformer_engine_layer_spec
from nemo.collections import llm
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from nemo.collections.llm.gpt.model.llama import LlamaConfig
import nemo_run as run
from nemo.collections.llm.recipes.finetune_default import nemo_resume
from nemo.lightning import OptimizerModule, io, teardown
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.llm.utils import Config
from transformers import CohereConfig as HFCohereConfig, CohereForCausalLM
from nemo.lightning.io.state import TransformFns, _ModelState
from nemo.lightning.pytorch.utils import dtype_from_hf
from megatron.core.models.gpt.gpt_model import GPTModel as MCoreGPTModel
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.enums import AttnBackend

class IdentityDetupleOp(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        if isinstance(x, tuple): 
            return x[0]
        return x

class IdentityDetupleFuncOp(IdentityDetupleOp):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        return super().forward


class CohereTransformerLayer(TransformerLayer):
    def __init__(self, config, submodules, *args, **kwargs):
        assert not config.add_bias_linear
        assert not config.add_qkv_bias
        assert not config.attention_dropout

        super().__init__(config, submodules, *args, **kwargs)
        self.cohere_input_layernorm = build_module(
            submodules.cohere_input_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.cohere_input_layernorm.bias.requires_grad_(False)
        
    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        # Copied from TransformerLayer.forward and modified to include cohere_input_layernorm
        # Remove 'dynamic_inference_decode_only' from kwargs if present
        # this is only used to uniquely identify decode and non-decode cuda graph
        # runners in the cuda graph manager
        kwargs.pop("dynamic_inference_decode_only", None)
        if self.recompute_input_layernorm:
            self.cohere_input_layernorm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            input_layernorm_output = self.cohere_input_layernorm_checkpoint.checkpoint(self.cohere_input_layernorm, hidden_states)
        else:
            input_layernorm_output = self.cohere_input_layernorm(hidden_states)

        hidden_states_attention, context = self._forward_attention(input_layernorm_output, *args, **kwargs)
        hidden_states_mlp = self._forward_mlp(input_layernorm_output, kwargs.get("inference_context", None))

        # we can theoretically fuse the bias_dropout_add here as well, but since attention_dropout is set to 0 and bias to False, we'll just add them regularly
        output = hidden_states + hidden_states_attention + hidden_states_mlp
        return output, context

@dataclass
class CohereTransformerLayerSubmodules(TransformerLayerSubmodules):
    cohere_input_layernorm: Union[ModuleSpec, type] = IdentityOp

def cohere_layer_spec(config: "GPTConfig") -> ModuleSpec:
    backend = TESpecProvider()
    spec = transformer_engine_layer_spec(config)
    spec.submodules = CohereTransformerLayerSubmodules(**spec.submodules.__dict__, cohere_input_layernorm=backend.layer_norm())
    spec.module = CohereTransformerLayer
    spec.submodules.self_attn_bda = IdentityDetupleFuncOp # residuals are applied in our custom class
    spec.submodules.mlp_bda = IdentityDetupleFuncOp # residuals are applied in our custom class
    spec.submodules.self_attention.submodules.linear_qkv = backend.column_parallel_linear()
    spec.submodules.mlp.submodules.linear_fc1 = backend.column_parallel_linear()
    spec.submodules.sharded_state_dict_keys_map.pop('mlp.0.weight')
    spec.submodules.sharded_state_dict_keys_map.pop('mlp.0.bias')
    return spec

class CohereMCoreGPTModel(MCoreGPTModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logit_scale = self.config.logit_scale
        if self.decoder.final_layernorm:
            self.decoder.final_layernorm.bias.requires_grad_(False)
    
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def _postprocess(self, **kwargs):
        ret = super()._postprocess(**kwargs)
        if not self.post_process:
            return ret
        # if we computed loss - we scaled there, otherwise scale here
        if kwargs.get('labels', None) is None:
            ret = ret * self.logit_scale
        
        return ret

    def compute_language_model_loss(self, labels: Tensor, logits: Tensor) -> Tensor:
        # scale the logits before computing loss
        scaled_logits = logits * self.logit_scale
        return super().compute_language_model_loss(labels, scaled_logits)


@dataclass
class CohereConfig(LlamaConfig):
    # configs that are common across model sizes
    normalization: str = "LayerNorm"
    add_bias_linear: bool = False
    add_qkv_bias: bool = False # in HF it's a config, but both 8B and 32B are set to False
    share_embeddings_and_output_weights: bool = True # default to True
    qk_layernorm: bool = False
    init_method_std: float = 0.02
    vocab_size: int = 256000
    layernorm_epsilon: float = 1e-5
    logit_scale: float = 0.125
    rotary_interleaved: bool = True
    transformer_layer_spec: Union[ModuleSpec, Callable[["GPTConfig"], ModuleSpec]] = cohere_layer_spec

    def configure_model(self, tokenizer, pre_process=None, post_process=None, vp_stage=None) -> "MCoreGPTModel":
        """Configure and instantiate a Cohere-Megatron Core GPT model based on this configuration."""

        assert not (HAVE_TE and self.use_transformer_engine_full_layer_spec), "We didn't copy this code from the original configure model so it isn't supported"
        assert self.attention_backend != AttnBackend.local, "We didn't copy this code from the original configure model so it isn't supported"
        assert not self.enable_cuda_graph, "We didn't copy this code from the original configure model so it isn't supported"
        assert not self.init_model_with_meta_device, "We didn't copy this code from the original configure model so it isn't supported"
        
        vp_size = self.virtual_pipeline_model_parallel_size
        is_pipeline_asymmetric = getattr(self, "account_for_embedding_in_pipeline_split", False) or getattr(self, "account_for_loss_in_pipeline_split", False)
        is_pipeline_asymmetric |= (getattr(self, "num_layers_in_first_pipeline_stage", None) or getattr(self, "num_layers_in_last_pipeline_stage", None)) is not None
        is_flexible_pp_layout = is_pipeline_asymmetric or (getattr(self, "pipeline_model_parallel_layout", None) is not None )
        if vp_size and not is_flexible_pp_layout:
            p_size = self.pipeline_model_parallel_size
            assert (self.num_layers // p_size) % vp_size == 0, "Make sure the number of model chunks is the same across all pipeline stages."

        # During fake lightning initialization, pass 0 to bypass the assertion that vp_stage must be
        # non-None when using virtual pipeline model parallelism
        vp_stage = vp_stage or 0

        transformer_layer_spec = self.transformer_layer_spec(self) # we know it's a callable in this config
        vocab_size = self.vocab_size
        if tokenizer is not None:
            logging.info(f"Use preset vocab_size: {vocab_size}, original vocab_size: {tokenizer.vocab_size}, dummy tokens: {vocab_size - tokenizer.vocab_size}.")

        return CohereMCoreGPTModel(
            self,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=self.seq_length,
            fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
            parallel_output=self.parallel_output,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            position_embedding_type=self.position_embedding_type,
            rotary_percent=self.rotary_percent,
            rotary_base=self.rotary_base,
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            pre_process=pre_process or parallel_state.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vp_stage),
            post_process=post_process or parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage),
            scatter_embedding_sequence_parallel=self.scatter_embedding_sequence_parallel,
            vp_stage=vp_stage
        )


@dataclass
class AyaExpanseConfig8B(CohereConfig):
    hidden_size: int = 4096
    ffn_hidden_size: int = 14336
    num_attention_heads: int = 32
    num_layers: int = 32
    num_query_groups: int = 8
    rotary_base: float = 10_000
    logit_scale: float = 0.125
    
@dataclass
class AyaExpanseConfig32B(CohereConfig):
    hidden_size: int = 8192
    ffn_hidden_size: int = 24576
    num_attention_heads: int = 64
    num_layers: int = 40
    num_query_groups: int = 8
    rotary_base: float = 4_000_000
    logit_scale: float = 0.0625


class CohereModel(GPTModel):
    """Cohere model implementation based on the GPT model architecture."""
    def __init__(
        self,
        config: Annotated[Optional[CohereConfig], Config[CohereConfig]] = None,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
        model_context_managers: Optional[List] = [],
    ):
        super().__init__(
            config or CohereConfig(),
            optim=optim,
            tokenizer=tokenizer,
            model_transform=model_transform,
            model_context_managers=model_context_managers,
        )

@io.model_importer(CohereModel, "hf")
class HFCohereImporter(io.ModelConnector["CohereForCausalLM", CohereModel]):
    def init(self) -> CohereModel:
        return CohereModel(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:
        from transformers import AutoConfig, AutoModelForCausalLM

        hf_config = AutoConfig.from_pretrained(str(self))
        source = AutoModelForCausalLM.from_pretrained(str(self), torch_dtype='auto')

        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        print(f"Converted Cohere model to Nemo, model saved to {output_path} in {source.dtype}.")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):
        mapping = {
            "model.embed_tokens.weight": "embedding.word_embeddings.weight",
            "model.layers.*.self_attn.o_proj.weight": "decoder.layers.*.self_attention.linear_proj.weight",
            "model.layers.*.input_layernorm.weight": "decoder.layers.*.cohere_input_layernorm.weight",
            "model.layers.*.input_layernorm.bias": "decoder.layers.*.cohere_input_layernorm.bias",
            "model.layers.*.mlp.down_proj.weight": "decoder.layers.*.mlp.linear_fc2.weight",
            "model.norm.weight": "decoder.final_layernorm.weight",
            "model.norm.bias": "decoder.final_layernorm.bias",
            "lm_head.weight": "output_layer.weight",
        }
        if getattr(source.config, "use_qk_norm", False):
            mapping["model.layers.*.self_attn.q_norm.weight"] = "decoder.layers.*.self_attention.q_layernorm.weight"
            mapping["model.layers.*.self_attn.k_norm.weight"] = "decoder.layers.*.self_attention.k_layernorm.weight"
        if getattr(source.config, "tie_word_embeddings", False):
            del mapping["lm_head.weight"]
        
        # populate the state dict with empty biases for layernorms
        sd = source.state_dict()
        for k in list(sd.keys()):
            if k.endswith('norm.weight'):
                sd[k.replace('.weight', '.bias')] = torch.zeros_like(sd[k])
        source_ = _ModelState(sd, source.config)
        
        transforms = [
            io.state_transform(
                source_key=(
                    "model.layers.*.self_attn.q_proj.weight",
                    "model.layers.*.self_attn.k_proj.weight",
                    "model.layers.*.self_attn.v_proj.weight",
                ),
                target_key="decoder.layers.*.self_attention.linear_qkv.weight",
                fn=TransformFns.merge_qkv,
            ),
            io.state_transform(
                source_key=("model.layers.*.mlp.gate_proj.weight", "model.layers.*.mlp.up_proj.weight"),
                target_key="decoder.layers.*.mlp.linear_fc1.weight",
                fn=TransformFns.merge_fc1,
            )
        ]

        return io.apply_transforms(source_, target, mapping=mapping, transforms=transforms)

    @property
    def tokenizer(self) -> "AutoTokenizer":
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
        return AutoTokenizer(self.save_hf_tokenizer_assets(str(self)))

    @property
    def config(self) -> CohereConfig:
        from transformers import AutoConfig, GenerationConfig

        source = AutoConfig.from_pretrained(str(self))
        try:
            generation_config = GenerationConfig.from_pretrained(str(self))
        except Exception:
            generation_config = None

        def make_vocab_size_divisible_by(vocab_size):
            base = 128
            while vocab_size % base != 0:
                base //= 2
            return base

        output = CohereConfig(
            add_qkv_bias=source.attention_bias,
            attention_dropout=source.attention_dropout,
            hidden_size=source.hidden_size,
            init_method_std=source.initializer_range,
            ffn_hidden_size=source.intermediate_size,
            layernorm_epsilon=source.layer_norm_eps,
            logit_scale=source.logit_scale,
            seq_length=source.max_position_embeddings,
            num_attention_heads=source.num_attention_heads,
            num_layers=source.num_hidden_layers,
            num_query_groups=source.num_key_value_heads,
            rotary_base=source.rope_theta,
            qk_layernorm=source.use_qk_norm,
            make_vocab_size_divisible_by=make_vocab_size_divisible_by(source.vocab_size),
            share_embeddings_and_output_weights=getattr(source, "tie_word_embeddings", False),
            fp16=(dtype_from_hf(source) == torch.float16),
            bf16=(dtype_from_hf(source) == torch.bfloat16),
            params_dtype=dtype_from_hf(source),
            generation_config=generation_config,
            vocab_size=source.vocab_size,
        )

        return output


@io.model_exporter(CohereModel, "hf")
class HFCohereExporter(io.ModelConnector[CohereModel, "CohereForCausalLM"]):
    def init(self, dtype=torch.bfloat16) -> "CohereForCausalLM":
        from transformers import AutoModelForCausalLM
        from transformers.modeling_utils import no_init_weights

        with no_init_weights():
            return AutoModelForCausalLM.from_config(self.config, torch_dtype=dtype)

    def apply(self, output_path: Path) -> Path:

        source, _ = self.nemo_load(str(self))
        source_config = source.config
        target = self.init(torch_dtype_from_mcore_config(source_config))
        target = self.convert_state(source, target, source_config)

        target = target.cpu()
        if self.config.tie_word_embeddings:
            state_dict = target.state_dict()
            state_dict.pop("lm_head.weight")
            target.save_pretrained(output_path, state_dict=state_dict)
        else:
            target.save_pretrained(output_path)

        try:
            self.tokenizer.tokenizer.save_pretrained(output_path)
        except Exception:
            logging.warning("Failed to save tokenizer")

        return output_path

    def convert_state(self, source, target, source_config=None):
        mapping = {
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.cohere_input_layernorm.weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            "decoder.final_layernorm.weight": "model.norm.weight"
        }
        if source.config.qk_layernorm:
            mapping["decoder.layers.*.self_attention.q_layernorm.weight"] = "model.layers.*.self_attn.q_norm.weight"
            mapping["decoder.layers.*.self_attention.k_layernorm.weight"] = "model.layers.*.self_attn.k_norm.weight"
        
        transforms = [
            io.state_transform(
                source_key="decoder.layers.*.self_attention.linear_qkv.weight",
                target_key=(
                    "model.layers.*.self_attn.q_proj.weight",
                    "model.layers.*.self_attn.k_proj.weight",
                    "model.layers.*.self_attn.v_proj.weight",
                ),
                fn=TransformFns.split_qkv,
            ),
            io.state_transform(
                source_key="decoder.layers.*.mlp.linear_fc1.weight",
                target_key=("model.layers.*.mlp.gate_proj.weight", "model.layers.*.mlp.up_proj.weight"),
                fn=TransformFns.split_fc1,
            ),
            io.state_transform(
                source_key="embedding.word_embeddings.weight",
                target_key="model.embed_tokens.weight",
                fn=TransformFns.prune_padding,
            ),
        ]
        if not source.config.share_embeddings_and_output_weights:
            transforms.append(
                io.state_transform(
                    source_key="output_layer.weight",
                    target_key="lm_head.weight",
                    fn=TransformFns.prune_padding,
                )
            )

        return io.apply_transforms(
            source,
            target,
            mapping=mapping,
            transforms=transforms,
        )

    @property
    def tokenizer(self) -> "TokenizerSpec":
        return io.load_context(str(self), subpath="model").tokenizer

    @property
    def config(self) -> "HFCohereConfig":
        source: CohereConfig = io.load_context(str(self), subpath="model.config")

        from transformers import CohereConfig as HFCohereConfig
        return HFCohereConfig(
            architectures=["CohereForCausalLM"],
            attention_bias=source.add_qkv_bias,
            attention_dropout=source.attention_dropout,
            hidden_size=source.hidden_size,
            initializer_range=source.init_method_std,
            intermediate_size=source.ffn_hidden_size,
            layer_norm_eps=source.layernorm_epsilon,
            logit_scale=source.logit_scale,
            max_position_embeddings=source.seq_length,
            num_attention_heads=source.num_attention_heads,
            num_hidden_layers=source.num_layers,
            num_key_value_heads=source.num_query_groups,
            rope_theta=source.rotary_base,
            use_qk_norm=source.qk_layernorm,
            tie_word_embeddings=source.share_embeddings_and_output_weights,
            vocab_size=source.vocab_size,
            bos_token_id=self.tokenizer.bos_id,
            eos_token_id=self.tokenizer.eos_id,
        )


############## RECIPE SECTION #################
class aya_expanse_8b:
    @staticmethod
    def pretrain_recipe(*args, **kwargs):
        recipe = llm.qwen3_8b.pretrain_recipe(*args, **kwargs)
        recipe.model = run.Config(CohereModel, config=run.Config(AyaExpanseConfig8B))
        return recipe
    
    @staticmethod
    def finetune_recipe(*args, **kwargs):
        recipe = llm.qwen3_8b.finetune_recipe(*args, **kwargs)
        recipe.model = run.Config(CohereModel, config=run.Config(AyaExpanseConfig8B))
        recipe.resume = nemo_resume('CohereLabs/aya-expanse-8b')
        return recipe

class aya_expanse_32b:
    @staticmethod
    def pretrain_recipe(*args, **kwargs):
        recipe = llm.qwen3_32b.pretrain_recipe(*args, **kwargs)
        recipe.model = run.Config(CohereModel, config=run.Config(AyaExpanseConfig32B))
        return recipe
    
    @staticmethod
    def finetune_recipe(*args, **kwargs):
        recipe = llm.qwen3_32b.finetune_recipe(*args, **kwargs)
        recipe.model = run.Config(CohereModel, config=run.Config(AyaExpanseConfig32B))
        recipe.resume = nemo_resume('CohereLabs/aya-expanse-32b')
        return recipe

class aya_llm:
    aya_expanse_8b = aya_expanse_8b
    aya_expanse_32b = aya_expanse_32b