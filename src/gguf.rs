use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q2_K,
    Q3_K,
    Q8_0,
    Q8_1,
    Q2_Kv2,
    Q3_Kv2,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    IQ2_XXS,
    IQ2_XS,
    IQ3_XXS,
    IQ1_S,
    IQ4_NL,
    IQ3_S,
    IQ2_S,
    IQ4_XS,
    I8,
    I16,
    I32,
    I64,
    F64,
    IQ1_M,
    BF16,
    TQ1_0,
    TQ2_0,
    MXFP4,
    NVFP4,
    Q1_0,
}

impl TensorType {
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => TensorType::F32,
            1 => TensorType::F16,
            2 => TensorType::Q4_0,
            3 => TensorType::Q4_1,
            4 => TensorType::Q5_0,
            5 => TensorType::Q5_1,
            6 => TensorType::Q2_K,
            7 => TensorType::Q3_K,
            8 => TensorType::Q8_0,
            9 => TensorType::Q8_1,
            10 => TensorType::Q2_K,
            11 => TensorType::Q3_K,
            12 => TensorType::Q4_K,
            13 => TensorType::Q5_K,
            14 => TensorType::Q6_K,
            15 => TensorType::Q8_K,
            16 => TensorType::IQ2_XXS,
            17 => TensorType::IQ2_XS,
            18 => TensorType::IQ3_XXS,
            19 => TensorType::IQ1_S,
            20 => TensorType::IQ4_NL,
            21 => TensorType::IQ3_S,
            22 => TensorType::IQ2_S,
            23 => TensorType::IQ4_XS,
            24 => TensorType::I8,
            25 => TensorType::I16,
            26 => TensorType::I32,
            27 => TensorType::I64,
            28 => TensorType::F64,
            29 => TensorType::IQ1_M,
            30 => TensorType::BF16,
            34 => TensorType::TQ1_0,
            35 => TensorType::TQ2_0,
            39 => TensorType::MXFP4,
            40 => TensorType::NVFP4,
            41 => TensorType::Q1_0,
            _ => panic!("unknown tensor type: {}", v),
        }
    }

    pub fn block_size(&self) -> usize {
        match self {
            TensorType::F32 | TensorType::I32 => 4,
            TensorType::F16 | TensorType::BF16 | TensorType::I16 => 2,
            TensorType::I8 => 1,
            TensorType::F64 | TensorType::I64 => 8,
            TensorType::Q4_0 | TensorType::Q4_1 => 1536,
            TensorType::Q5_0 => 264,
            TensorType::Q5_1 => 24,
            TensorType::Q8_0 => 34,
            TensorType::Q8_1 => 35,
            TensorType::Q2_K => 16,
            TensorType::Q3_K => 112,
            TensorType::Q4_K => 144,
            TensorType::Q5_K => 162,
            TensorType::Q6_K => 210,
            TensorType::Q8_K => 144,
            TensorType::IQ2_XXS => 16,
            TensorType::IQ2_XS => 256,
            TensorType::IQ3_XXS => 16,
            TensorType::IQ1_S => 32,
            TensorType::IQ4_NL => 16,
            TensorType::IQ3_S => 16,
            TensorType::IQ2_S => 256,
            TensorType::IQ4_XS => 16,
            TensorType::TQ1_0 => 256,
            TensorType::TQ2_0 => 34,
            TensorType::MXFP4 => 832,
            TensorType::NVFP4 => 832,
            TensorType::Q1_0 => 256,
            TensorType::IQ1_M => 512,
        }
    }

    pub fn block_elements(&self) -> usize {
        let total_bytes = self.block_size();
        match self {
            TensorType::F32 | TensorType::I32 => 1,
            TensorType::F16 | TensorType::BF16 | TensorType::I16 => 1,
            TensorType::I8 => 1,
            TensorType::F64 | TensorType::I64 => 1,
            TensorType::Q4_0 | TensorType::Q4_1 => 16,
            TensorType::Q5_0 => 32,
            TensorType::Q5_1 => 32,
            TensorType::Q8_0 => 32,
            TensorType::Q8_1 => 32,
            TensorType::Q2_K | TensorType::Q3_K | TensorType::Q4_K | TensorType::Q5_K | TensorType::Q6_K | TensorType::Q8_K => 256,
            TensorType::IQ2_XXS | TensorType::IQ2_XS | TensorType::IQ3_XXS | TensorType::IQ4_XS | TensorType::IQ3_S => 256,
            TensorType::IQ1_S | TensorType::IQ4_NL | TensorType::IQ1_M | TensorType::Q1_0 => 256,
            TensorType::IQ2_S | TensorType::TQ1_0 | TensorType::TQ2_0 | TensorType::MXFP4 | TensorType::NVFP4 => 256,
        }
    }

    pub fn is_supported(&self) -> bool {
        matches!(
            self,
            TensorType::F32
                | TensorType::F16
                | TensorType::BF16
                | TensorType::Q8_0
                | TensorType::Q4_K
        )
    }

    pub fn is_quantized(&self) -> bool {
        matches!(
            self,
            TensorType::Q4_0
                | TensorType::Q4_1
                | TensorType::Q5_0
                | TensorType::Q5_1
                | TensorType::Q8_0
                | TensorType::Q8_1
                | TensorType::Q2_K
                | TensorType::Q3_K
                | TensorType::Q4_K
                | TensorType::Q5_K
                | TensorType::Q6_K
                | TensorType::Q8_K
        )
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: TensorType,
    pub data_offset: u64,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.shape.iter().product()
    }

    pub fn n_blocks(&self) -> usize {
        let n = self.n_elements() as usize;
        let b = self.dtype.block_elements();
        if b == 0 {
            return 0;
        }
        n.div_ceil(b)
    }

    pub fn n_bytes(&self) -> u64 {
        let blocks = self.n_blocks();
        if blocks == 0 {
            return self.n_elements() * self.dtype.block_size() as u64;
        }
        (blocks * self.dtype.block_size()) as u64
    }
}

#[derive(Debug, Clone)]
pub enum GGUFValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GGUFValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl GGUFValue {
    pub fn as_string(&self) -> Option<&str> {
        match self {
            GGUFValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GGUFValue::U32(v) => Some(*v as u64),
            GGUFValue::U64(v) => Some(*v),
            GGUFValue::I64(v) => Some(*v as u64),
            GGUFValue::I32(v) => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[GGUFValue]> {
        match self {
            GGUFValue::Array(arr) => Some(arr),
            _ => None,
        }
    }
}

#[derive(Debug, Default)]
pub struct GGUFMetadata {
    pub architecture: Option<String>,
    pub general_name: Option<String>,
    pub block_count: Option<u64>,
    pub context_length: Option<u64>,
    pub embedding_length: Option<u64>,
    pub feed_forward_length: Option<u64>,
    pub attention_head_count: Option<u64>,
    pub attention_head_count_kv: Option<u64>,
    pub attention_key_length: Option<u64>,
    pub attention_value_length: Option<u64>,
    pub attention_layer_norm_rms_epsilon: Option<f32>,
    pub rope_freq_base: Option<f32>,
    pub rope_dimension_count: Option<u64>,
    pub rope_dimension_count_swa: Option<u64>,
    pub rope_freq_base_swa: Option<f32>,
    pub rope_scaling_type: Option<String>,
    pub rope_scaling_factor: Option<f32>,
    pub rope_scaling_original_context_length: Option<u64>,
    pub attention_sliding_window: Option<u64>,
    pub attention_sliding_window_pattern: Option<Vec<u64>>,
    pub expert_count: Option<u64>,
    pub expert_used_count: Option<u64>,
    pub expert_feed_forward_length: Option<u64>,
    pub expert_shared_feed_forward_length: Option<u64>,
    pub expert_gating_func: Option<u32>,
    pub expert_weights_norm: Option<bool>,
    pub expert_weights_scale: Option<f32>,
    pub leading_dense_block_count: Option<u64>,
    pub vocab_size: Option<u64>,
    pub file_type: Option<u32>,
    pub target_layers: Option<Vec<i64>>,
    pub decoder_arch: Option<String>,
    pub alignment: Option<u64>,
    pub raw_fields: Vec<(String, GGUFValue)>,
}

pub struct GGUFReader {
    pub file: File,
    pub version: u32,
    pub tensor_count: u64,
    pub field_count: u64,
    pub tensors: Vec<TensorInfo>,
    pub metadata: GGUFMetadata,
    pub tensor_data_offset: u64,
}

impl GGUFReader {
    pub fn open(path: &Path) -> std::io::Result<Self> {
        let mut file = File::open(path)?;

        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if &magic != b"GGUF" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "not a GGUF file",
            ));
        }

        let version = read_u32(&mut file)?;
        if version != 3 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("only GGUF v3 is supported (found v{})", version),
            ));
        }
        let tensor_count = read_u64(&mut file)?;
        let field_count = read_u64(&mut file)?;

        let mut metadata = GGUFMetadata::default();
        let mut raw_fields = Vec::with_capacity(field_count as usize);

        for _ in 0..field_count {
            let name_len = read_u64(&mut file)?;
            let mut name_bytes = vec![0u8; name_len as usize];
            file.read_exact(&mut name_bytes)?;
            let name = String::from_utf8_lossy(&name_bytes).into_owned();

            let type_id = read_u32(&mut file)?;
            let value = read_value(&mut file, type_id)?;

            raw_fields.push((name.clone(), value.clone()));
            populate_metadata(&mut metadata, &name, &value);
        }

        metadata.raw_fields = raw_fields;

        let alignment = metadata.alignment.unwrap_or(32).max(1);

        let mut tensors = Vec::with_capacity(tensor_count as usize);

        for _ in 0..tensor_count {
            let name_len = read_u64(&mut file)?;
            let mut name_bytes = vec![0u8; name_len as usize];
            file.read_exact(&mut name_bytes)?;
            let name = String::from_utf8_lossy(&name_bytes).into_owned();

            let n_dims = read_u32(&mut file)?;
            let mut shape = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                shape.push(read_u64(&mut file)?);
            }

            let dtype_id = read_u32(&mut file)?;
            let dtype = TensorType::from_u32(dtype_id);

            let data_offset = read_u64(&mut file)?;

            tensors.push(TensorInfo {
                name,
                shape,
                dtype,
                data_offset,
            });
        }

        let mut tensor_data_offset = file.stream_position()?;
        if alignment > 1 {
            tensor_data_offset = (tensor_data_offset + alignment - 1) & !(alignment - 1);
            file.seek(SeekFrom::Start(tensor_data_offset))?;
        }

        Ok(GGUFReader {
            file,
            version,
            tensor_count,
            field_count,
            tensors,
            metadata,
            tensor_data_offset,
        })
    }

    pub fn tensor_by_name(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    pub fn tensors_with_dtype(&self, dtype: TensorType) -> Vec<&TensorInfo> {
        self.tensors.iter().filter(|t| t.dtype == dtype).collect()
    }

    pub fn read_tensor_data(&mut self, tensor: &TensorInfo) -> std::io::Result<Vec<u8>> {
        let abs_offset = self.tensor_data_offset + tensor.data_offset;
        self.file.seek(SeekFrom::Start(abs_offset))?;
        let mut buf = vec![0u8; tensor.n_bytes() as usize];
        self.file.read_exact(&mut buf)?;
        Ok(buf)
    }

    pub fn layer_count(&self) -> Option<u64> {
        self.metadata.block_count
    }

    pub fn embedding_length(&self) -> Option<u64> {
        self.metadata.embedding_length
    }
}

fn read_u32(file: &mut File) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(file: &mut File) -> std::io::Result<u64> {
    let mut buf = [0u8; 8];
    file.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_value(file: &mut File, type_id: u32) -> std::io::Result<GGUFValue> {
    match type_id {
        0 => {
            let mut buf = [0u8; 1];
            file.read_exact(&mut buf)?;
            Ok(GGUFValue::Uint8(buf[0]))
        }
        1 => {
            let mut buf = [0u8; 1];
            file.read_exact(&mut buf)?;
            Ok(GGUFValue::Int8(buf[0] as i8))
        }
        2 => {
            let mut buf = [0u8; 2];
            file.read_exact(&mut buf)?;
            Ok(GGUFValue::Uint16(u16::from_le_bytes(buf)))
        }
        3 => {
            let mut buf = [0u8; 2];
            file.read_exact(&mut buf)?;
            Ok(GGUFValue::Int16(i16::from_le_bytes(buf)))
        }
        4 => Ok(GGUFValue::U32(read_u32(file)?)),
        5 => {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf)?;
            Ok(GGUFValue::I32(i32::from_le_bytes(buf)))
        }
        6 => {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf)?;
            Ok(GGUFValue::F32(f32::from_le_bytes(buf)))
        }
        7 => {
            let mut buf = [0u8; 1];
            file.read_exact(&mut buf)?;
            Ok(GGUFValue::Bool(buf[0] != 0))
        }
        8 => {
            let len = read_u64(file)?;
            let mut buf = vec![0u8; len as usize];
            file.read_exact(&buf)?;
            let s = String::from_utf8_lossy(&buf).into_owned();
            Ok(GGUFValue::String(s))
        }
        9 => {
            let array_type_id = read_u32(file)?;
            let array_len = read_u64(file)?;
            let mut elements = Vec::with_capacity(array_len as usize);
            for _ in 0..array_len {
                elements.push(read_value(file, array_type_id)?);
            }
            Ok(GGUFValue::Array(elements))
        }
        10 => Ok(GGUFValue::U64(read_u64(file)?)),
        11 => {
            let mut buf = [0u8; 8];
            file.read_exact(&mut buf)?;
            Ok(GGUFValue::I64(i64::from_le_bytes(buf)))
        }
        12 => {
            let mut buf = [0u8; 8];
            file.read_exact(&mut buf)?;
            Ok(GGUFValue::F64(f64::from_le_bytes(buf)))
        }
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("unknown GGUF value type: {}", type_id),
        )),
    }
}

fn populate_metadata(metadata: &mut GGUFMetadata, key: &str, value: &GGUFValue) {
    match key {
        "general.architecture" => metadata.architecture = value.as_string().map(String::from),
        "general.name" => metadata.general_name = value.as_string().map(String::from),
        "general.file_type" => metadata.file_type = value.as_u64().map(|v| v as u32),
        "general.alignment" => metadata.alignment = value.as_u64(),
        "laguna.block_count" | "dflash.block_count" | "block_count" => {
            metadata.block_count = value.as_u64();
        }
        "laguna.context_length" | "dflash.context_length" | "context_length" => {
            metadata.context_length = value.as_u64();
        }
        "laguna.embedding_length" | "dflash.embedding_length" | "embedding_length" => {
            metadata.embedding_length = value.as_u64();
        }
        "laguna.feed_forward_length" | "dflash.feed_forward_length" | "feed_forward_length" => {
            metadata.feed_forward_length = value.as_u64();
        }
        "laguna.attention.head_count" | "dflash.attention.head_count" | "attention.head_count" => {
            if let Some(arr) = value.as_array() {
                if let Some(first) = arr.first() {
                    metadata.attention_head_count = first.as_u64();
                }
            } else {
                metadata.attention_head_count = value.as_u64();
            }
        }
        "laguna.attention.head_count_kv" | "dflash.attention.head_count_kv" | "attention.head_count_kv" => {
            if let Some(arr) = value.as_array() {
                if arr.len() > 1 {
                    metadata.attention_head_count_kv = arr[1].as_u64();
                }
            } else {
                metadata.attention_head_count_kv = value.as_u64();
            }
        }
        "laguna.attention.key_length" | "dflash.attention.key_length" => {
            metadata.attention_key_length = value.as_u64();
        }
        "laguna.attention.value_length" | "dflash.attention.value_length" => {
            metadata.attention_value_length = value.as_u64();
        }
        "laguna.attention.layer_norm_rms_epsilon"
        | "dflash.attention.layer_norm_rms_epsilon" => {
            metadata.attention_layer_norm_rms_epsilon = value.as_u64().map(|v| v as f32);
        }
        "laguna.rope.freq_base" | "dflash.rope.freq_base" | "rope.freq_base" => {
            metadata.rope_freq_base = value.as_u64().map(|v| v as f32);
        }
        "laguna.rope.dimension_count"
        | "dflash.rope.dimension_count"
        | "rope.dimension_count" => {
            metadata.rope_dimension_count = value.as_u64();
        }
        "laguna.rope.dimension_count_swa"
        | "dflash.rope.dimension_count_swa"
        | "rope.dimension_count_swa" => {
            metadata.rope_dimension_count_swa = value.as_u64();
        }
        "laguna.attention.sliding_window" | "dflash.attention.sliding_window" => {
            metadata.attention_sliding_window = value.as_u64();
        }
        "laguna.attention.sliding_window_pattern"
        | "dflash.attention.sliding_window_pattern" => {
            if let Some(arr) = value.as_array() {
                metadata.attention_sliding_window_pattern =
                    Some(arr.iter().filter_map(|v| v.as_u64().map(|u| u as u64)).collect());
            }
        }
        "expert.count" | "expert_count" => {
            metadata.expert_count = value.as_u64();
        }
        "expert.used_count" | "expert_used_count" => {
            metadata.expert_used_count = value.as_u64();
        }
        "expert.feed_forward_length" | "expert_feed_forward_length" => {
            metadata.expert_feed_forward_length = value.as_u64();
        }
        "expert.shared_feed_forward_length" | "expert_shared_feed_forward_length" => {
            metadata.expert_shared_feed_forward_length = value.as_u64();
        }
        "expert.gating_func" | "expert_gating_func" => {
            metadata.expert_gating_func = value.as_u64().map(|v| v as u32);
        }
        "vocab_size" => {
            metadata.vocab_size = value.as_u64();
        }
        "general.vocab_size" => {
            metadata.vocab_size = value.as_u64();
        }
        "dflash.target_layers" | "laguna.target_layers" | "target_layers" => {
            if let Some(arr) = value.as_array() {
                metadata.target_layers =
                    Some(arr.iter().filter_map(|v| v.as_u64().map(|u| u as i64)).collect());
            }
        }
        "dflash.decoder_arch" | "decoder_arch" => {
            metadata.decoder_arch = value.as_string().map(String::from);
        }
        _ => {}
    }
    if let Some(arr) = value.as_array() {
        if let Some(first) = arr.first() {
            if let Some(u) = first.as_u64() {
                if key.ends_with(".rope_dimension_count") && key.contains("general.") {
                    metadata.rope_dimension_count = Some(u);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_elements() {
        assert_eq!(TensorType::F32.block_elements(), 1);
        assert_eq!(TensorType::Q8_0.block_elements(), 32);
        assert_eq!(TensorType::Q4_K.block_elements(), 256);
    }

    #[test]
    fn test_supported_types() {
        assert!(TensorType::F32.is_supported());
        assert!(TensorType::F16.is_supported());
        assert!(TensorType::BF16.is_supported());
        assert!(TensorType::Q8_0.is_supported());
        assert!(TensorType::Q4_K.is_supported());
        assert!(!TensorType::Q5_0.is_supported());
    }

    #[test]
    fn test_block_sizes() {
        assert_eq!(TensorType::F32.block_size(), 4);
        assert_eq!(TensorType::F16.block_size(), 2);
        assert_eq!(TensorType::BF16.block_size(), 2);
        assert_eq!(TensorType::Q8_0.block_size(), 34);
        assert_eq!(TensorType::Q4_K.block_size(), 144);
    }

    #[test]
    fn test_tensor_n_bytes_bf16() {
        let t = TensorInfo {
            name: "test".into(),
            shape: vec![100, 200],
            dtype: TensorType::BF16,
            data_offset: 0,
        };
        assert_eq!(t.n_elements(), 20000);
        assert_eq!(t.n_bytes(), 40000);
    }

    #[test]
    fn test_tensor_n_bytes_f32() {
        let t = TensorInfo {
            name: "test".into(),
            shape: vec![100, 200],
            dtype: TensorType::F32,
            data_offset: 0,
        };
        assert_eq!(t.n_elements(), 20000);
        assert_eq!(t.n_bytes(), 80000);
    }

    #[test]
    fn test_tensor_n_bytes_q8_0() {
        let t = TensorInfo {
            name: "test".into(),
            shape: vec![1024],
            dtype: TensorType::Q8_0,
            data_offset: 0,
        };
        assert_eq!(t.n_elements(), 1024);
        assert_eq!(t.n_bytes(), 1120);
    }

    #[test]
    fn test_tensor_n_bytes_q4_k_3d() {
        let t = TensorInfo {
            name: "test".into(),
            shape: vec![256, 10, 3],
            dtype: TensorType::Q4_K,
            data_offset: 0,
        };
        assert_eq!(t.n_elements(), 7680);
        assert_eq!(t.n_blocks(), 31);
        assert_eq!(t.n_bytes(), 31 * 144);
    }

    #[test]
    fn test_tensor_type_from_u32() {
        assert_eq!(TensorType::from_u32(0), TensorType::F32);
        assert_eq!(TensorType::from_u32(8), TensorType::Q8_0);
        assert_eq!(TensorType::from_u32(12), TensorType::Q4_K);
        assert_eq!(TensorType::from_u32(30), TensorType::BF16);
    }

    #[test]
    fn test_value_as_string() {
        let v = GGUFValue::String("hello".into());
        assert_eq!(v.as_string(), Some("hello"));
        let v = GGUFValue::U32(42);
        assert_eq!(v.as_string(), None);
    }

    #[test]
    fn test_value_as_u64() {
        let v = GGUFValue::U32(42);
        assert_eq!(v.as_u64(), Some(42));
        let v = GGUFValue::U64(100);
        assert_eq!(v.as_u64(), Some(100));
        let v = GGUFValue::String("hello".into());
        assert_eq!(v.as_u64(), None);
    }
}
