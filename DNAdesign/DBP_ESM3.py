import contextlib
import os
import sys
from peft import LoraConfig, get_peft_model
import attr
import torch
from esm.models.esm3 import ESM3, ESMOutput
import torch.nn as nn
from esm.sdk.api import ESMProteinTensor, LogitsConfig, LogitsOutput, ForwardTrackData, ESMProtein, ProteinType, \
    GenerationConfig, ESM3InferenceClient, ESMProteinError, SamplingTrackConfig, SamplingConfig
from esm.utils.constants import esm3 as C
from esm.utils.generation import _get_non_special_tokens, _get_masked_positions, _stack_protein_tensors, \
    _slice_tensor_dataclass, _trim_sequence_tensor_dataclass, _get_annealed_temperature, _sample_per_prompt, \
    _get_iterative_sampling_mask_for_prompt_and_step
from esm.utils.sampling import _BatchedESMProteinTensor

from esm.utils.structure.affine3d import (
    build_affine3d_from_coordinates,
)
from typing import Callable, Sequence, List

from esm.models.esm3 import ESM3
from peft import PeftModel
from esm.models.function_decoder import FunctionTokenDecoder
from esm.models.vqvae import (
    StructureTokenDecoder,
    StructureTokenEncoder,
)
from esm.tokenization import (
    get_esm3_model_tokenizers,
    TokenizerCollectionProtocol,
)
from esm.utils.constants.esm3 import data_root
from esm.utils.constants.models import ESM3_OPEN_SMALL

from tqdm import tqdm

# ==============================================================================
# NOTE: iterative_sampling_raw, iterative_sampling_tokens, _batch_forward,
#       and the various ESM3 model builder functions (ESM3_structure_encoder_v0, etc.)
#       are mostly unchanged from your original code, as they handle the generation
#       loop, which calls the model's forward/logits methods that we are modifying.
#       I will keep them here for completeness.
# ==============================================================================

ModelBuilder = Callable[[torch.device | str], nn.Module]


def iterative_sampling_raw(
        DNA_embed: torch.Tensor,
        client: ESM3InferenceClient,
        proteins: list[ESMProtein],
        configs: list[GenerationConfig],
) -> list[ESMProtein | ESMProteinError]:
    input_tokens = [client.encode(protein) for protein in proteins]
    output_tokens_list = client.batch_generate(DNA_embed, input_tokens, configs)
    raw_proteins: list[ESMProtein | ESMProteinError] = []
    for output_tokens in output_tokens_list:
        if isinstance(output_tokens, ESMProteinTensor):
            raw_proteins.append(client.decode(output_tokens))
        elif isinstance(output_tokens, ESMProteinError):
            raw_proteins.append(output_tokens)
        else:
            raise ValueError(f"Unknown output type {type(output_tokens)}")

    for input_protein, raw_protein, config in zip(proteins, raw_proteins, configs):
        if isinstance(raw_protein, ESMProteinError):
            continue
        if config.track not in ["function", "residue_annotations"]:
            raw_protein.function_annotations = input_protein.function_annotations
    return raw_proteins


def iterative_sampling_tokens(
        DNA_embedd: torch.Tensor,
        client: ESM3InferenceClient,
        input_tokens: list[ESMProteinTensor],
        configs: list[GenerationConfig],
        tokenizers: TokenizerCollectionProtocol,
) -> Sequence[ESMProteinTensor | ESMProteinError]:
    devices = set([t.device for t in input_tokens])
    if len(devices) > 1:
        raise AttributeError(f"Input tokens on multiple devices {devices}")

    sampled_tokens = [attr.evolve(tokens) for tokens in input_tokens]

    for tokens, config in zip(sampled_tokens, configs):
        if config.condition_on_coordinates_only and tokens.coordinates is not None:
            tokens.structure = None

    sequence_lengths = [len(tokens) for tokens in sampled_tokens]
    total_to_sample = []
    for protein, config in zip(sampled_tokens, configs):
        track = config.track
        if getattr(protein, track) is None:
            num_sampling_steps = _get_non_special_tokens(protein, tokenizers)
        else:
            masked = _get_masked_positions(
                track, getattr(protein, track), getattr(tokenizers, track).mask_token_id
            )
            num_sampling_steps = torch.sum(masked).item()
        total_to_sample.append(num_sampling_steps)
        if (num_sampling_steps > 0) and (num_sampling_steps < config.num_steps):
            config.num_steps = int(num_sampling_steps)

    max_num_steps = max([config.num_steps for config in configs])
    batched_tokens = _stack_protein_tensors(
        sampled_tokens, sequence_lengths, tokenizers, devices.pop()
    )
    errors: dict[int, ESMProteinError] = {}
    disable_tqdm = bool(os.environ.get("DISABLE_ITERATIVE_SAMPLING_TQDM", False))
    for t in tqdm(range(max_num_steps), disable=disable_tqdm):
        forward_out = _batch_forward(DNA_embedd, client, batched_tokens)
        for i, config in enumerate(configs):  # B
            if i in errors:
                continue

            if config.track in ["coordinates", "residue_annotations"]:
                errors[i] = ESMProteinError(
                    error_code=500,
                    error_msg=f"Iterative sampling {config.track} is not supported.",
                )
                continue

            if t >= config.num_steps:
                continue

            per_prompt_cur_sampled = _BatchedESMProteinTensor.from_protein_tensor(
                batched_tokens.slice(i)
            )
            per_prompt_forward_out: LogitsOutput = _slice_tensor_dataclass(
                forward_out, i, keep_dim=True
            )
            per_prompt_forward_out = _trim_sequence_tensor_dataclass(
                per_prompt_forward_out,
                len(per_prompt_cur_sampled),
            )

            if config.temperature_annealing:
                temperature = _get_annealed_temperature(
                    t, config.num_steps, config.temperature
                )
            else:
                temperature = config.temperature

            track_sample_config = SamplingTrackConfig()
            track_sample_config.invalid_ids = config.invalid_ids
            track_sample_config.temperature = temperature
            track_sample_config.top_p = config.top_p
            sampling_config = SamplingConfig(**{config.track: track_sample_config})
            per_prompt_forward_and_sample_output = _sample_per_prompt(
                per_prompt_cur_sampled,
                per_prompt_forward_out,
                sampling_config,
                tokenizers,
                decode_sasa_tokens=False,
            )
            per_prompt_new_sampled = per_prompt_forward_and_sample_output.protein_tensor
            assert per_prompt_forward_and_sample_output.entropy is not None
            try:
                where_to_sample = _get_iterative_sampling_mask_for_prompt_and_step(
                    per_prompt_cur_sampled,
                    torch.tensor(sequence_lengths[i]),
                    torch.tensor(total_to_sample[i]),
                    t,
                    per_prompt_forward_and_sample_output.entropy,
                    config,
                    tokenizers,
                )
            except ValueError as e:
                errors[i] = ESMProteinError(error_code=500, error_msg=str(e))
                continue

            where_to_sample.to(input_tokens[0].device)
            old_track_samples = getattr(per_prompt_cur_sampled, config.track)
            new_track_samples = getattr(per_prompt_new_sampled, config.track)
            new_track_samples = torch.where(
                where_to_sample, new_track_samples, old_track_samples
            )
            getattr(batched_tokens, config.track)[i, ...] = new_track_samples[0]

    output_tokens = [
        batched_tokens.slice(i, sequence_len=sequence_lengths[i])
        if i not in errors
        else errors[i]
        for i in range(len(input_tokens))
    ]
    for inputs, outputs, config in zip(input_tokens, output_tokens, configs):
        if isinstance(outputs, ESMProteinError):
            continue
        setattr(outputs, "coordinates", getattr(inputs, "coordinates"))
        for f in attr.fields(SamplingConfig):
            if "embedding" in f.name or f.name == "return_hidden_states":
                continue
            if f.name != config.track:
                setattr(outputs, f.name, getattr(inputs, f.name))
    return output_tokens


def _batch_forward(DNA_embedd: torch.Tensor, client: ESM3InferenceClient, protein: _BatchedESMProteinTensor):
    return client.logits(
        DNA_embedd,
        protein,
        LogitsConfig(
            sequence=True, structure=True, secondary_structure=True, sasa=True,
            function=True, residue_annotations=True, return_embeddings=True,
        ),
    )


def ESM3_structure_encoder_v0(device: torch.device | str = "cpu"):
    with torch.device(device):
        model = StructureTokenEncoder(d_model=1024, n_heads=1, v_heads=128, n_layers=2, d_out=128, n_codes=4096).eval()
    state_dict = torch.load(data_root("esm3") / "data/weights/esm3_structure_encoder_v0.pth", map_location=device)
    model.load_state_dict(state_dict)
    return model


def ESM3_structure_decoder_v0(device: torch.device | str = "cpu"):
    with torch.device(device):
        model = StructureTokenDecoder(d_model=1280, n_heads=20, n_layers=30).eval()
    state_dict = torch.load(data_root("esm3") / "data/weights/esm3_structure_decoder_v0.pth", map_location=device)
    model.load_state_dict(state_dict)
    return model


def ESM3_function_decoder_v0(device: torch.device | str = "cpu"):
    with torch.device(device):
        model = FunctionTokenDecoder().eval()
    state_dict = torch.load(data_root("esm3") / "data/weights/esm3_function_decoder_v0.pth", map_location=device)
    model.load_state_dict(state_dict)
    return model


# ==============================================================================
# NEW: Helper Modules for EiRA-ESM3
# ==============================================================================
class DNATransformer(nn.Module):
    """
    A dedicated Transformer Encoder to process and project DNA embeddings.
    """

    def __init__(self, dna_emb_dim: int, nhead: int, num_encoder_layers: int, dim_feedforward: int,
                 dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dna_emb_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            src: Raw DNA embeddings (B, L_dna, D_dna)
            src_key_padding_mask: Mask for padding in DNA sequence (B, L_dna)
        Returns:
            Projected DNA embeddings (B, L_dna, D_dna)
        """
        return self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)


class CrossAttentionInjectionBlock(nn.Module):
    """
    This block performs cross-attention between protein and DNA embeddings,
    followed by a residual connection and a feed-forward network.
    It's injected after a standard ESM3 Transformer layer.
    """

    def __init__(self, d_model: int, dna_emb_dim: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            kdim=dna_emb_dim,
            vdim=dna_emb_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        self.gate_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, protein_hidden_state: torch.Tensor, dna_embedding: torch.Tensor,
                dna_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            protein_hidden_state: Output from an ESM3 self-attention block (B, L_prot, D_prot). This is the Query.
            dna_embedding: Projected DNA embeddings (B, L_dna, D_dna). This is the Key and Value.
            dna_padding_mask: Mask for padding in DNA sequence (B, L_dna).
        """
        # 1. Cross-Attention + Residual
        attn_output, _ = self.cross_attn(
            query=protein_hidden_state,
            key=dna_embedding,
            value=dna_embedding,
            key_padding_mask=dna_padding_mask
        )

        gate = self.gate_proj(protein_hidden_state)  # 动态门控信号
        gated_attn = gate * attn_output  # 控制DNA信息强度
        x = protein_hidden_state + self.dropout1(gated_attn)  # 保护原始表示
        # x = protein_hidden_state + self.dropout1(attn_output)
        x = self.norm1(x)

        # 2. FFN + Residual
        ffn_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)
        return x


# ==============================================================================
# NEW: EiRA-ESM3 Model Implementation
# ==============================================================================

class EiRA_ESM3(ESM3):
    def __init__(self,
                 # ESM3 Base arguments
                 d_model=1536,
                 n_heads=24,
                 v_heads=256,
                 n_layers=48,
                 structure_encoder_fn=ESM3_structure_encoder_v0,
                 structure_decoder_fn=ESM3_structure_decoder_v0,
                 function_decoder_fn=ESM3_function_decoder_v0,
                 tokenizers=get_esm3_model_tokenizers(ESM3_OPEN_SMALL),

                 # EiRA specific arguments
                 dna_emb_dim: int = 4096,
                 dna_transformer_layers: int = 2,
                 dna_transformer_heads: int = 8,
                 injection_layers: int = 4,  # Number of final ESM3 layers to inject cross-attention
                 cross_attention_heads: int = 8,
                 device="cpu",
                 **esm3_kwargs):

        # Initialize the base ESM3 model
        super().__init__(d_model, n_heads, v_heads, n_layers,
                         structure_encoder_fn, structure_decoder_fn,
                         function_decoder_fn, tokenizers, **esm3_kwargs)

        self.injection_layers = injection_layers
        assert injection_layers <= n_layers, "Cannot inject into more layers than exist in ESM3."

        # 1. DNA Transformer for projecting DNA embeddings
        self.dna_transformer = DNATransformer(
            dna_emb_dim=dna_emb_dim,
            nhead=dna_transformer_heads,
            num_encoder_layers=dna_transformer_layers,
            dim_feedforward=dna_emb_dim * 4,  # Common practice for FFN dimension
            dropout=0.1
        )

        # 2. A list of Cross-Attention blocks to be injected
        #    We create one for each injection layer.
        self.cross_attention_injectors = nn.ModuleList([
            CrossAttentionInjectionBlock(
                d_model=d_model,
                dna_emb_dim=dna_emb_dim,
                nhead=cross_attention_heads,
                dim_feedforward=d_model * 4,  # FFN dimension inside the injection block
                dropout=0.1
            ) for _ in range(injection_layers)
        ])

        # Load pretrained weights for the base ESM3 part
        # Note: The new modules (dna_transformer, cross_attention_injectors) will have random weights
        # and need to be trained.
        print("Loading pretrained ESM3 weights...")
        state_dict = torch.load(
            data_root("esm3") / "data/weights/esm3_sm_open_v1.pth", map_location=device
        )
        load_result = self.load_state_dict(state_dict, strict=False)
        print("--- EiRA Model Initialization ---")
        print(f"New modules not in pretrained checkpoint (should be trained): {load_result.missing_keys}")
        print(f"Pretrained weights not used (should be empty): {load_result.unexpected_keys}")
        print("---------------------------------")

    def forward(
            self,
            *,
            # New required inputs for EiRA
            dna_embeddings: torch.Tensor,  # Shape: (B, L_dna, D_dna)
            dna_padding_mask: torch.Tensor | None = None,  # Shape: (B, L_dna)

            # Standard ESM3 inputs
            sequence_tokens: torch.Tensor | None = None,
            structure_tokens: torch.Tensor | None = None,
            ss8_tokens: torch.Tensor | None = None,
            sasa_tokens: torch.Tensor | None = None,
            function_tokens: torch.Tensor | None = None,
            residue_annotation_tokens: torch.Tensor | None = None,
            average_plddt: torch.Tensor | None = None,
            per_res_plddt: torch.Tensor | None = None,
            structure_coords: torch.Tensor | None = None,
            chain_id: torch.Tensor | None = None,
            sequence_id: torch.Tensor | None = None,
    ) -> ESMOutput:

        # --- Input Preprocessing (Identical to base ESM3) ---
        # This part ensures all protein-related tensors are correctly shaped and padded.
        if not hasattr(self, 'tokenizers') or self.tokenizers is None:
            raise AttributeError("Tokenizers not found in ESM3 base class.")

        protein_input_present = any(x is not None for x in [
            sequence_tokens, structure_tokens, ss8_tokens, sasa_tokens,
            structure_coords, function_tokens, residue_annotation_tokens,
        ])
        if protein_input_present:
            L, device = next((x.shape[1], x.device) for x in [
                sequence_tokens, structure_tokens, ss8_tokens, sasa_tokens,
                structure_coords, function_tokens, residue_annotation_tokens,
            ] if x is not None)
        elif sequence_tokens is not None:
            L, device = sequence_tokens.shape[1], sequence_tokens.device
        else:
            raise ValueError("At least one protein-related input must be non-None.")

        t = self.tokenizers
        defaults = lambda x, tok: torch.full((dna_embeddings.size(0), L), tok, dtype=torch.long,
                                             device=device) if x is None else x

        sequence_tokens = defaults(sequence_tokens, t.sequence.mask_token_id)
        # ... (the rest of the default tensor creation from your code)
        ss8_tokens = defaults(ss8_tokens, C.SS8_PAD_TOKEN)
        sasa_tokens = defaults(sasa_tokens, C.SASA_PAD_TOKEN)
        average_plddt = defaults(average_plddt, 1).float()
        per_res_plddt = defaults(per_res_plddt, 0).float()
        chain_id = defaults(chain_id, 0)

        # ... (rest of preprocessing logic is complex and should be kept as is)
        if structure_coords is None:
            structure_coords = torch.full((dna_embeddings.size(0), L, 3, 3), float("nan"), dtype=torch.float,
                                          device=device)
        structure_coords = structure_coords[..., :3, :]
        affine, affine_mask = build_affine3d_from_coordinates(structure_coords)

        structure_tokens = defaults(structure_tokens, C.STRUCTURE_MASK_TOKEN)
        assert structure_tokens is not None
        # ... (token masking logic)
        structure_tokens = (structure_tokens.masked_fill(structure_tokens == -1, C.STRUCTURE_MASK_TOKEN)
                            .masked_fill(sequence_tokens == C.SEQUENCE_BOS_TOKEN, C.STRUCTURE_BOS_TOKEN)
                            .masked_fill(sequence_tokens == C.SEQUENCE_PAD_TOKEN, C.STRUCTURE_PAD_TOKEN)
                            .masked_fill(sequence_tokens == C.SEQUENCE_EOS_TOKEN, C.STRUCTURE_EOS_TOKEN)
                            .masked_fill(sequence_tokens == C.SEQUENCE_CHAINBREAK_TOKEN, C.STRUCTURE_CHAINBREAK_TOKEN)
                            )
        if residue_annotation_tokens is None:
            residue_annotation_tokens = torch.full((dna_embeddings.size(0), L, 16), C.RESIDUE_PAD_TOKEN,
                                                   dtype=torch.long, device=device)
        if function_tokens is None:
            function_tokens = torch.full((dna_embeddings.size(0), L, 8), C.INTERPRO_PAD_TOKEN, dtype=torch.long,
                                         device=device)

        # --- EiRA Forward Pass ---

        # 1. Get initial protein embeddings from the ESM3 encoder
        protein_embeddings = self.encoder(
            sequence_tokens, structure_tokens, average_plddt, per_res_plddt,
            ss8_tokens, sasa_tokens, function_tokens, residue_annotation_tokens,
        )

        # 2. Project DNA embeddings using the dedicated DNA transformer
        projected_dna_embeddings = self.dna_transformer(dna_embeddings, dna_padding_mask)

        # 3. Manually iterate through the transformer layers and inject cross-attention
        hidden_states = protein_embeddings
        all_hidden_states = []

        # FIX: Access the transformer layers via `self.transformer.blocks` instead of `.layers`
        num_esm_layers = len(self.transformer.blocks)
        injection_start_layer = num_esm_layers - self.injection_layers

        # FIX: Iterate over `self.transformer.blocks`
        for i, layer in enumerate(self.transformer.blocks):
            all_hidden_states.append(hidden_states)

            # Pass through the standard ESM3 transformer layer
            hidden_states = layer(hidden_states, sequence_id, affine, affine_mask, chain_id)

            # If we are in the injection window, apply the cross-attention block
            if i >= injection_start_layer:
                injector_idx = i - injection_start_layer
                injector = self.cross_attention_injectors[injector_idx]
                hidden_states = injector(
                    protein_hidden_state=hidden_states,
                    dna_embedding=projected_dna_embeddings,
                    dna_padding_mask=dna_padding_mask
                )

        # After the loop, `hidden_states` is the pre-LayerNorm output of the final block
        embedding = hidden_states
        # FIX: Apply the final layer norm from the base ESM3 transformer, which is named `norm`
        x = self.transformer.norm(hidden_states)

        # 4. Pass the final DNA-conditioned representation to the output heads
        return self.output_heads(x, embedding)

    # --- The logits() and generate() methods can be inherited or slightly adapted ---
    # The key change is ensuring `dna_embedding` is passed to `forward`.

    def logits(
            self,
            dna_embedding: torch.Tensor,
            input: ESMProteinTensor | _BatchedESMProteinTensor,
            config: LogitsConfig = LogitsConfig(),
    ) -> LogitsOutput:
        if not isinstance(input, _BatchedESMProteinTensor):
            input = _BatchedESMProteinTensor.from_protein_tensor(input)

        device = torch.device(input.device)

        if input.coordinates is None:
            per_res_plddt = None
        else:
            per_res_plddt = input.coordinates.isfinite().all(dim=-1).any(dim=-1).float()

        with (
            torch.no_grad(),
            torch.autocast(enabled=True, device_type=device.type, dtype=torch.bfloat16)
            if device.type == "cuda"
            else contextlib.nullcontext(),
        ):
            output = self.forward(
                dna_embeddings=dna_embedding.to(device),  # Make sure DNA embedding is on the right device
                # dna_padding_mask can be added here if you have one
                sequence_tokens=input.sequence,
                structure_tokens=input.structure,
                ss8_tokens=input.secondary_structure,
                sasa_tokens=input.sasa,
                function_tokens=input.function,
                residue_annotation_tokens=input.residue_annotations,
                average_plddt=torch.tensor(1.0, device=input.device),
                per_res_plddt=per_res_plddt,
                structure_coords=input.coordinates,
                chain_id=None,
                sequence_id=None,
            )

        output = ESMOutput(
            **{k: v.to(device).to(torch.float32) for k, v in vars(output).items()}
        )

        return LogitsOutput(
            logits=ForwardTrackData(
                sequence=output.sequence_logits if config.sequence else None,
                structure=output.structure_logits if config.structure else None,
                secondary_structure=output.secondary_structure_logits if config.secondary_structure else None,
                sasa=output.sasa_logits if config.sasa else None,
                function=output.function_logits if config.function else None,
            ),
            residue_annotation_logits=output.residue_logits if config.residue_annotations else None,
            embeddings=output.embeddings if config.return_embeddings else None,
        )

    def generate(self, DNA_embedding: torch.Tensor, input: ProteinType, config: GenerationConfig) -> ProteinType:
        proteins = self.batch_generate(DNA_embedding, [input], [config])
        assert len(proteins) == 1
        return proteins[0]

    def batch_generate(
            self, DNA_embeds: torch.Tensor, inputs: list[ProteinType], configs: list[GenerationConfig]
    ) -> list[ProteinType]:
        assert len(inputs) == len(configs), "Must have the same number of prompts and configs."
        if not inputs:
            return []

        t = type(inputs[0])
        for i in range(1, len(inputs)):
            assert isinstance(inputs[i], t)

        # These iterative sampling functions will call the `.logits()` method we defined above
        if isinstance(inputs[0], ESMProtein):
            return iterative_sampling_raw(DNA_embeds, self, inputs, configs)  # type: ignore
        elif isinstance(inputs[0], ESMProteinTensor):
            return iterative_sampling_tokens(
                DNA_embeds, self, inputs, configs, self.tokenizers,  # type: ignore
            )
        else:
            raise ValueError("Input must be an ESMProtein or ESMProteinTensor")


if __name__ == '__main__':
    # Use a smaller model for faster testing if possible, but stick to the specified arch
    # The parameters d_model, n_heads, n_layers must match the pretrained weights being loaded.
    D_MODEL = 1536
    N_HEADS = 24
    N_LAYERS = 48
    DNA_EMB_DIM = 4096  # Your specified DNA embedding dimension
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print(f"Initializing EiRA_ESM3 model on {DEVICE}...")

    # Instantiate the new model
    model = EiRA_ESM3(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dna_emb_dim=DNA_EMB_DIM,
        injection_layers=4,  # Inject into the last 4 layers
        dna_transformer_layers=2,  # Use 2 layers for the DNA transformer
        device=DEVICE
    ).to(DEVICE)

    model=PeftModel.from_pretrained(model,
                                    r"E:\science\EiRA\model_check_point\EiRA_checkpoint_vanilla_lora_ft32_repeat_penalty"
                                    ,is_trainable=True)

    # --- Training Setup Example (using LoRA or full fine-tuning) ---

    # Example of making only specific new parts trainable (full fine-tuning)
    # 1. Freeze the entire model first
    fi=open('./all_DBP_EiRA_parameters.txt', 'w')
    for name, param in model.named_parameters():
        fi.write(name+'\n')
        if param.requires_grad:print(name)
    fi.close()

    sys.exit(0)


    # 2. Unfreeze the newly added modules
    for param in model.dna_transformer.parameters():
        param.requires_grad = True
    for param in model.cross_attention_injectors.parameters():
        param.requires_grad = True


    print("\nTrainable parameters (full fine-tuning of new parts):")
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            print(f"- {name:<60} Shape: {param.shape}")
            trainable_params += param.numel()
    print(f"Total trainable parameters: {trainable_params / 1e6:.2f}M")
    print(f"Total model parameters: {total_params / 1e6:.2f}M")

    # --- Inference Example ---
    model.eval()
    print("\nRunning a test generation...")
    with torch.no_grad():
        # Create a dummy DNA embedding
        # Batch size = 1, DNA length = 50, DNA dimension = 4096
        dummy_dna_embedding = torch.randn(1, 50, DNA_EMB_DIM).to(DEVICE)

        # Define the protein to generate
        input_protein = ESMProtein(sequence='MT______GLLK__________________YSPOLL')

        # Define generation config
        gen_config = GenerationConfig(track='sequence', num_steps=10)  # 10 steps to fill 10 masks

        # Generate
        res_protein = model.generate(
            DNA_embedding=dummy_dna_embedding,
            input=input_protein,
            config=gen_config
        )

        print("\n--- Generation Result ---")
        print(f"Input Sequence:  {input_protein.sequence}")
        print(f"Output Sequence: {res_protein.sequence}")
        # Note: plddt and ptm might be lower quality without a proper structure track
        if res_protein.plddt is not None:
            print(f"Mean pLDDT: {res_protein.plddt.mean().item():.2f}")
        if res_protein.ptm is not None:
            print(f"PTM: {res_protein.ptm.item():.2f}")
        print("-------------------------\n")

