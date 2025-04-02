use crate::layers::HiddenAct;
use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct NVEmbedConfig {
    // Primary model config
    pub hidden_size: usize,
    pub model_type: String,
    pub torch_dtype: Option<String>,
    pub padding_side: Option<String>,
    pub is_mask_instruction: Option<bool>,
    pub add_pad_token: Option<bool>,
    pub add_eos: Option<bool>,
    pub mask_type: Option<String>,
    
    // Text config (bidir_mistral)
    #[serde(rename = "text_config")]
    pub text_config: TextConfig,
    
    // Latent attention config
    #[serde(rename = "latent_attention_config")]
    pub latent_attention_config: LatentAttentionConfig,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub hidden_act: HiddenAct,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub sliding_window: Option<usize>,
    pub vocab_size: usize,
    pub model_type: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct LatentAttentionConfig {
    pub model_type: String,
    pub num_latents_value: usize,
    pub num_cross_heads: usize,
    pub output_normalize: bool,
    pub hidden_dim: usize,
    pub latent_dim: usize,
    pub cross_dim_head: usize,
}