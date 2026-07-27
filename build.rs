//! Build script: compiles GLSL shaders to SPIR-V and embeds them as byte arrays.
//!
//! Shaders are in `kernels/*.comp`. Each produces a `kernels/<name>.spv` file
//! that is embedded into the binary via `include_bytes!`.
//!
//! Requires `glslangValidator` on PATH (see scripts/setup-dev-env.sh).

use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;

fn main() {
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    let kernels_dir = Path::new("kernels");
    let out_dir_path = Path::new(&out_dir);

    fs::create_dir_all(out_dir_path).expect("failed to create OUT_DIR");

    let shaders = [
        "enc_aux_norm.comp",
        "dequant.comp",
        "dequant_bf16.comp",
        "dequant_q4k.comp",
        "rmsnorm_rope.comp",
        "rmsnorm.comp",
        "rope.comp",
        "attention.comp",
        "softmax.comp",
        "ff_moe.comp",
        "activation.comp",
        "gate_mul.comp",
        "add.comp",
        "proj.comp",
        "repeat_v.comp",
        "qk_norm.comp",
        "matmul.comp",
        "sampler.comp",
        "laguna_layer_fused.comp",
    ];

    for shader in &shaders {
        let src = kernels_dir.join(shader);
        if !src.exists() {
            continue;
        }

        let dst = out_dir_path.join(format!("{}.spv", shader));
        let status = Command::new("glslangValidator")
            .args(["-V", "--target-env", "vulkan1.2", "-o"])
            .arg(&dst)
            .arg(&src)
            .status();

        match status {
            Ok(s) if s.success() => {
                println!("cargo:rerun-if-changed={}", src.display());
            }
            Ok(s) => {
                panic!(
                    "glslangValidator failed for {shader} (exit {:?}): {s}",
                    s.code()
                );
            }
            Err(e) => {
                panic!(
                    "failed to run glslangValidator for {}: {}. \
                     Run scripts/setup-dev-env.sh to install it.",
                    shader, e
                );
            }
        }
    }

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    println!("cargo:rustc-env=LAGUNA_SHADER_DIR={}", manifest_dir);
}
