use crate::flash_attn::flash_attn_varlen;
use crate::layers::{get_cos_sin, get_inv_freqs, HiddenAct, Linear, RMSNorm};
use crate::models::{Model, NVEmbedConfig};
use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module, VarBuilder, LayerNorm};
use candle_rotary::apply_rotary_inplace;
use text_embeddings_backend_core::{Batch, ModelType, Pool};

// ================ BidirectionalMistral Implementation ================

struct BidirectionalMistralAttention {
    qkv_linear: Linear,
    o_proj: Linear,

    window_size_left: Option<usize>,

    num_attention_heads: usize,
    num_key_value_heads: usize,
    attention_head_size: usize,

    softmax_scale: f32,

    span: tracing::Span,
}

impl BidirectionalMistralAttention {
    pub fn load(vb: VarBuilder, config: &NVEmbedConfig) -> Result<Self> {
        let window_size_left = config.text_config.sliding_window;
        let num_attention_heads = config.text_config.num_attention_heads;
        let attention_head_size = config.text_config.hidden_size / config.text_config.num_attention_heads;
        let num_key_value_heads = config.text_config.num_key_value_heads;
        let hidden_size = config.text_config.hidden_size;

        let query_weight = vb.pp("q_proj").get((hidden_size, hidden_size), "weight")?;

        let key_weight = vb.pp("k_proj").get(
            (num_key_value_heads * attention_head_size, hidden_size),
            "weight",
        )?;

        let value_weight = vb.pp("v_proj").get(
            (num_key_value_heads * attention_head_size, hidden_size),
            "weight",
        )?;

        let qkv_weight = Tensor::cat(&[&query_weight, &key_weight, &value_weight], 0)?;
        let qkv_linear = Linear::new(qkv_weight, None, None);

        let o_proj_weight = vb.pp("o_proj").get((hidden_size, hidden_size), "weight")?;

        let o_proj = Linear::new(o_proj_weight, None, None);

        let softmax_scale = (1. / (attention_head_size as f64).sqrt()) as f32;

        Ok(Self {
            qkv_linear,
            o_proj,
            window_size_left,
            num_attention_heads,
            num_key_value_heads,
            attention_head_size,
            softmax_scale,
            span: tracing::span!(tracing::Level::TRACE, "attention"),
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        cu_seqlens: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        max_s: usize,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let qkv = self.qkv_linear.forward(hidden_states)?;

        // Reshape to [tokens, heads, head_size]
        let mut new_qkv_shape = qkv.dims().to_vec();
        new_qkv_shape.pop();
        new_qkv_shape.push(self.num_attention_heads + 2 * self.num_key_value_heads);
        new_qkv_shape.push(self.attention_head_size);

        let qkv = qkv.reshape(new_qkv_shape)?;

        // Split qkv tensor
        let q = qkv.narrow(1, 0, self.num_attention_heads)?;
        let k = qkv.narrow(1, self.num_attention_heads, self.num_key_value_heads)?;
        let v = qkv.narrow(
            1,
            self.num_attention_heads + self.num_key_value_heads,
            self.num_key_value_heads,
        )?;

        apply_rotary_inplace(&q, &k, &cos, &sin, true)?;

        // Note: is_causal is set to false for bidirectional attention
        let attention = flash_attn_varlen(
            &q,
            &k,
            &v,
            None,
            cu_seqlens,
            cu_seqlens,
            max_s,
            max_s,
            self.softmax_scale,
            false, // is_causal=false for bidirectional
            self.window_size_left,
        )?;
        let attention = attention.flatten_from(candle::D::Minus2)?;

        self.o_proj.forward(&attention)
    }
}

struct BidirectionalMistralMLP {
    gate_up_proj: Linear,
    down_proj: Linear,

    act: HiddenAct,
    intermediate_size: usize,

    span: tracing::Span,
}

impl BidirectionalMistralMLP {
    pub fn load(vb: VarBuilder, config: &NVEmbedConfig) -> Result<Self> {
        let intermediate_size = config.text_config.intermediate_size;
        let hidden_size = config.text_config.hidden_size;

        let gate_proj_weight = vb
            .pp("gate_proj")
            .get((intermediate_size, hidden_size), "weight")?;

        let up_proj_weight = vb
            .pp("up_proj")
            .get((intermediate_size, hidden_size), "weight")?;

        let gate_up_proj_weight = Tensor::cat(&[&gate_proj_weight, &up_proj_weight], 0)?;
        let gate_up_proj = Linear::new(gate_up_proj_weight, None, None);

        let down_proj_weight = vb
            .pp("down_proj")
            .get((hidden_size, intermediate_size), "weight")?;
        let down_proj = Linear::new(down_proj_weight, None, None);

        Ok(Self {
            gate_up_proj,
            down_proj,
            intermediate_size,
            act: config.text_config.hidden_act.clone(),
            span: tracing::span!(tracing::Level::TRACE, "mlp"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let gate_up_states = self.gate_up_proj.forward(hidden_states)?;
        let gate_states = gate_up_states.narrow(1, 0, self.intermediate_size)?;
        let up_states = gate_up_states.narrow(1, self.intermediate_size, self.intermediate_size)?;

        let gate_states = match self.act {
            HiddenAct::Gelu => gate_states.gelu(),
            HiddenAct::Relu => gate_states.relu(),
            HiddenAct::Swiglu => gate_states.silu(),
        }?;
        let r = self.down_proj.forward(&(gate_states * up_states)?)?;
        Ok(r)
    }
}

struct BidirectionalMistralLayer {
    attention: BidirectionalMistralAttention,
    mlp: BidirectionalMistralMLP,
    input_layer_norm: RMSNorm,
    post_attention_layer_norm: RMSNorm,

    span: tracing::Span,
}

impl BidirectionalMistralLayer {
    pub fn load(vb: VarBuilder, config: &NVEmbedConfig) -> Result<Self> {
        let attention = BidirectionalMistralAttention::load(vb.pp("self_attn"), config)?;
        let mlp = BidirectionalMistralMLP::load(vb.pp("mlp"), config)?;

        let input_layer_norm = RMSNorm::load(
            vb.pp("input_layernorm"),
            config.text_config.hidden_size,
            config.text_config.rms_norm_eps,
        )?;
        let post_attention_layer_norm = RMSNorm::load(
            vb.pp("post_attention_layernorm"),
            config.text_config.hidden_size,
            config.text_config.rms_norm_eps,
        )?;

        Ok(Self {
            attention,
            mlp,
            input_layer_norm,
            post_attention_layer_norm,
            span: tracing::span!(tracing::Level::TRACE, "layer"),
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        residual: Option<&Tensor>,
        cu_seqlens: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        max_s: usize,
    ) -> Result<(Tensor, Tensor)> {
        let _enter = self.span.enter();

        let (normed_hidden_states, res) = self.input_layer_norm.forward(hidden_states, residual)?;
        let attn_output =
            self.attention
                .forward(&normed_hidden_states, cu_seqlens, cos, sin, max_s)?;
        let (normed_attn_res_output, attn_res) = self
            .post_attention_layer_norm
            .forward(&attn_output, Some(&res))?;
        let mlp_output = self.mlp.forward(&normed_attn_res_output)?;

        Ok((mlp_output, attn_res))
    }
}

struct BidirectionalMistralModel {
    embeddings: Embedding,
    layers: Vec<BidirectionalMistralLayer>,
    norm: RMSNorm,
    cos_cache: Tensor,
    sin_cache: Tensor,
    pub device: Device,
    
    span: tracing::Span,
}

impl BidirectionalMistralModel {
    pub fn load(vb: VarBuilder, config: &NVEmbedConfig) -> Result<Self> {
        match vb.device() {
            Device::Cuda(_) => {}
            _ => candle::bail!("BidirectionalMistral requires Cuda"),
        }

        if vb.dtype() != DType::F16 {
            candle::bail!("BidirectionalMistral requires DType::F16")
        }

        // Navigate to the text_encoder path if needed
        let has_text_encoder = vb.contains_tensor("text_encoder.embed_tokens.weight");
        let vb = if has_text_encoder {
            vb.pp("text_encoder")
        } else {
            vb
        };

        let embeddings = Embedding::new(
            vb.pp("embed_tokens")
                .get((config.text_config.vocab_size, config.text_config.hidden_size), "weight")?,
            config.text_config.hidden_size,
        );

        let layers = (0..config.text_config.num_hidden_layers)
            .map(|index| BidirectionalMistralLayer::load(vb.pp(format!("layers.{index}")), config))
            .collect::<Result<Vec<_>>>()?;

        let norm = RMSNorm::load(vb.pp("norm"), config.text_config.hidden_size, config.text_config.rms_norm_eps)?;

        let head_dim = config.text_config.hidden_size / config.text_config.num_attention_heads;
        let inv_freqs = get_inv_freqs(
            head_dim,
            config.text_config.rope_theta,
            vb.device(),
            None,
        )?;
        let (cos_cache, sin_cache) = get_cos_sin(
            config.text_config.max_position_embeddings,
            &inv_freqs,
            vb.dtype(),
            false, // Not use half
        )?;

        Ok(Self {
            embeddings,
            layers,
            norm,
            cos_cache,
            sin_cache,
            device: vb.device().clone(),
            span: tracing::span!(tracing::Level::TRACE, "bidir_mistral"),
        })
    }

    pub fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor, cu_seqlens: &Tensor, position_ids: &Tensor, max_s: usize) -> Result<Tensor> {
        let _enter = self.span.enter();

        // Get embeddings
        let mut hidden_states = self.embeddings.forward(input_ids)?;

        // Get cos and sin for rotary
        let cos = self.cos_cache.index_select(&position_ids, 0)?;
        let sin = self.sin_cache.index_select(&position_ids, 0)?;

        // Forward through layers
        let mut residual = None;
        for layer in &self.layers {
            let (h, r) = layer.forward(
                &hidden_states,
                residual.as_ref(),
                cu_seqlens,
                &cos,
                &sin,
                max_s,
            )?;
            hidden_states = h;
            residual = Some(r);
        }

        // Final normalization
        let (outputs, _) = self.norm.forward(&hidden_states, residual.as_ref())?;
        Ok(outputs)
    }
}

// ================ Latent Attention Implementation ================

// Implements the PreNorm wrapper from the Python code
struct PreNorm {
    norm: LayerNorm,
    context_norm: Option<LayerNorm>,
}

impl PreNorm {
    fn new(dim: usize, context_dim: Option<usize>) -> Result<Self> {
        let norm = LayerNorm::new(dim, 1e-5)?;
        let context_norm = if let Some(ctx_dim) = context_dim {
            Some(LayerNorm::new(ctx_dim, 1e-5)?)
        } else {
            None
        };
        
        Ok(Self {
            norm,
            context_norm,
        })
    }
    
    fn forward(&self, x: &Tensor, context: Option<&Tensor>) -> Result<(Tensor, Option<Tensor>)> {
        let normed_x = self.norm.forward(x)?;
        
        let normed_context = if let Some(ctx) = context {
            if let Some(ctx_norm) = &self.context_norm {
                Some(ctx_norm.forward(ctx)?)
            } else {
                Some(ctx.clone())
            }
        } else {
            None
        };
        
        Ok((normed_x, normed_context))
    }
}

// Implements the GEGLU activation from the Python code
struct GEGLU {}

impl GEGLU {
    fn forward(x: &Tensor) -> Result<Tensor> {
        let (x, gates) = x.chunk(2, -1)?;
        let gates_activated = gates.gelu()?;
        x * gates_activated
    }
}

// Implements the FeedForward layer from the Python code
struct FeedForward {
    net_w1: Linear,
    net_w2: Linear,
}

impl FeedForward {
    fn load(vb: VarBuilder, dim: usize, mult: usize) -> Result<Self> {
        let inner_dim = dim * mult;
        let w1 = vb.pp("net.0").get((dim, inner_dim * 2), "weight")?;
        let w2 = vb.pp("net.2").get((inner_dim, dim), "weight")?;
        
        let net_w1 = Linear::new(w1, None, None);
        let net_w2 = Linear::new(w2, None, None);
        
        Ok(Self {
            net_w1,
            net_w2,
        })
    }
    
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.net_w1.forward(x)?;
        let x = GEGLU::forward(&x)?;
        self.net_w2.forward(&x)
    }
}

// Implements the Attention in LatentAttention from the Python code
struct CrossAttention {
    to_q: Linear,
    to_kv: Linear,
    to_out: Linear,
    heads: usize,
    scale: f32,
}

impl CrossAttention {
    fn load(vb: VarBuilder, query_dim: usize, context_dim: usize, heads: usize, dim_head: usize) -> Result<Self> {
        let inner_dim = dim_head * heads;
        let scale = (1.0 / (dim_head as f64).sqrt()) as f32;
        
        let to_q = Linear::new(
            vb.pp("to_q").get((inner_dim, query_dim), "weight")?,
            None,
            None,
        );
        
        let to_kv = Linear::new(
            vb.pp("to_kv").get((inner_dim * 2, context_dim), "weight")?,
            None,
            None,
        );
        
        let to_out = Linear::new(
            vb.pp("to_out").get((query_dim, inner_dim), "weight")?,
            None,
            None,
        );
        
        Ok(Self {
            to_q,
            to_kv,
            to_out,
            heads,
            scale,
        })
    }
    
    fn forward(&self, x: &Tensor, context: &Tensor) -> Result<Tensor> {
        let q = self.to_q.forward(x)?;
        let kv = self.to_kv.forward(context)?;
        
        let (k, v) = kv.chunk(2, -1)?;
        
        // Reshape to separate the heads
        let (b, n, _) = x.dims3()?;
        let head_dim = q.dim(2)? / self.heads;
        
        let q = q.reshape((b, n, self.heads, head_dim))?.permute((0, 2, 1, 3))?; // [b, h, n, d]
        let k = k.reshape((b, k.dim(1)?, self.heads, head_dim))?.permute((0, 2, 1, 3))?; // [b, h, m, d]
        let v = v.reshape((b, v.dim(1)?, self.heads, head_dim))?.permute((0, 2, 1, 3))?; // [b, h, m, d]
        
        // Scaled dot-product attention
        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * self.scale)?;
        let attn_weights = attn_weights.softmax(3)?; // softmax along seq_len_k
        
        let context = attn_weights.matmul(&v)?; // [b, h, n, d]
        let context = context.permute((0, 2, 1, 3))?.reshape((b, n, -1))?; // [b, n, h*d]
        
        self.to_out.forward(&context)
    }
}

struct LatentAttentionModel {
    cross_attend_norm: PreNorm,
    cross_attention: CrossAttention,
    ff_norm: PreNorm,
    feed_forward: FeedForward,
    latents: Tensor, // Learned latent vectors
    output_normalize: bool,
    
    span: tracing::Span,
}

impl LatentAttentionModel {
    fn load(vb: VarBuilder, config: &NVEmbedConfig) -> Result<Self> {
        let la_config = &config.latent_attention_config;
        let latent_dim = la_config.latent_dim;
        let hidden_dim = la_config.hidden_dim;
        let cross_heads = la_config.num_cross_heads;
        let cross_dim_head = la_config.cross_dim_head / cross_heads;
        
        // Load latents parameter
        let latents = vb.get((la_config.num_latents_value, latent_dim), "latents")?;
        
        // Create cross attention components
        let cross_attend_norm = PreNorm::new(latent_dim, Some(hidden_dim))?;
        let cross_attention = CrossAttention::load(
            vb.pp("cross_attend_blocks.0.fn"), 
            latent_dim, 
            hidden_dim, 
            cross_heads, 
            cross_dim_head
        )?;
        
        // Create feed forward components
        let ff_norm = PreNorm::new(latent_dim, None)?;
        let feed_forward = FeedForward::load(
            vb.pp("cross_attend_blocks.1.fn"), 
            latent_dim, 
            4 // Multiplier for inner dimension
        )?;
        
        Ok(Self {
            cross_attend_norm,
            cross_attention,
            ff_norm,
            feed_forward,
            latents,
            output_normalize: la_config.output_normalize,
            span: tracing::span!(tracing::Level::TRACE, "latent_attention"),
        })
    }
    
    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let _enter = self.span.enter();
        
        let (batch_size, seq_len, _) = hidden_states.dims3()?;
        
        // Repeat latents for each item in batch
        let x = self.latents.repeat((batch_size, 1, 1))?;
        
        // Cross attention block
        let (normed_x, normed_context) = self.cross_attend_norm.forward(&x, Some(hidden_states))?;
        let cross_attn = self.cross_attention.forward(&normed_x, normed_context.as_ref().unwrap())?;
        let x = (x + cross_attn)?;
        
        // Feed forward block
        let (normed_x, _) = self.ff_norm.forward(&x, None)?;
        let ff = self.feed_forward.forward(&normed_x)?;
        let output = (x + ff)?;
        
        // Mean pooling with attention mask if provided
        if let Some(mask) = attention_mask {
            // Expand the mask to match hidden dimensions
            let mask_expanded = mask.unsqueeze(-1)?.to_dtype(self.latents.dtype()?)?;
            
            // Apply mask and compute masked mean
            let sum = (hidden_states * mask_expanded)?.sum(1)?;
            let mask_sum = mask.sum_keepdim(1)?.to_dtype(self.latents.dtype()?)?;
            let mean = (sum / mask_sum)?;
            
            // Normalize if required
            if self.output_normalize {
                let norm = mean.sqr()?.sum_keepdim(1)?.sqrt()?;
                (mean / norm)?
            } else {
                mean
            }
        } else {
            // Mean pooling without mask
            let mean = output.mean(1)?;
            
            // Normalize if required
            if self.output_normalize {
                let norm = mean.sqr()?.sum_keepdim(1)?.sqrt()?;
                (mean / norm)?
            } else {
                mean
            }
        }
    }
}

// ================ Main NVEmbed Model Implementation ================

pub struct FlashNVEmbedModel {
    embedding_model: BidirectionalMistralModel,
    latent_attention_model: LatentAttentionModel,
    is_mask_instruction: bool,
    pub device: Device,
    
    span: tracing::Span,
}

impl FlashNVEmbedModel {
    pub fn load(vb: VarBuilder, config: &NVEmbedConfig, model_type: ModelType) -> Result<Self> {
        match vb.device() {
            Device::Cuda(_) => {}
            _ => candle::bail!("FlashNVEmbed requires Cuda"),
        }

        if vb.dtype() != DType::F16 {
            candle::bail!("FlashNVEmbed requires DType::F16")
        }

        // Check model type
        match model_type {
            ModelType::Classifier => {
                candle::bail!("`classifier` model type is not supported for NVEmbed")
            }
            ModelType::Embedding(_) => {}
        };

        // Load the embedding model (BidirectionalMistral)
        let embedding_model = BidirectionalMistralModel::load(vb.clone(), config)?;
        
        // Load the latent attention model
        let latent_attention_model = LatentAttentionModel::load(
            vb.pp("latent_attention_model"), 
            config
        )?;
        
        // Default to true if not specified
        let is_mask_instruction = config.is_mask_instruction.unwrap_or(true);

        Ok(Self {
            embedding_model,
            latent_attention_model,
            is_mask_instruction,
            device: vb.device().clone(),
            span: tracing::span!(tracing::Level::TRACE, "nvembed"),
        })
    }

    pub fn forward(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let _enter = self.span.enter();

        let batch_size = batch.cumulative_seq_lengths.len() - 1;
        let shape = batch.input_ids.len();

        // Create Cuda tensors
        let input_ids = Tensor::from_vec(batch.input_ids, shape, &self.device)?;
        let position_ids = Tensor::from_vec(batch.position_ids, shape, &self.device)?;
        let attention_mask = Tensor::from_vec(
            batch.attention_mask.clone(),
            shape,
            &self.device,
        )?;
        let cu_seqlens = Tensor::from_vec(
            batch.cumulative_seq_lengths.clone(),
            batch_size + 1,
            &self.device,
        )?;

        // Create pool_mask - This would handle the instruction masking in a real implementation
        // For now, we'll use the normal attention mask since we don't have instruction tokens
        let pool_mask = attention_mask.clone();

        // Run the embedding model (BidirectionalMistral)
        let hidden_states = self.embedding_model.forward(
            &input_ids, 
            &attention_mask,
            &cu_seqlens,
            &position_ids, 
            batch.max_length as usize,
        )?;

        // Run the latent attention model to get embeddings
        let embeddings = self.latent_attention_model.forward(
            &hidden_states, 
            Some(&pool_mask)
        )?;

        // Return the embeddings for all pooled indices
        if !batch.pooled_indices.is_empty() {
            let indices = Tensor::from_vec(
                batch.pooled_indices.clone(), 
                batch.pooled_indices.len(), 
                &self.device
            )?;
            
            // For NVEmbed, we use the single embedding for each sequence
            let selected_embeddings = embeddings.index_select(&indices, 0)?;
            
            Ok((Some(selected_embeddings), None))
        } else {
            Ok((None, None))
        }
    }
}

impl Model for FlashNVEmbedModel {
    fn is_padded(&self) -> bool {
        false
    }
    
    fn embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        self.forward(batch)
    }
}