use std::env;
use std::path::Path;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("usage: {} <command> [args...]", args[0]);
        eprintln!("commands:");
        eprintln!("  info <gguf_path>          - show model metadata");
        eprintln!("  shapes <gguf_path>        - show tensor shapes");
        eprintln!("  forward <gguf_path> <tok> - run forward pass");
        eprintln!("  test-dequant              - test dequantization");
        eprintln!("  test-matmul               - test matmul on GPU");
        eprintln!("  test-rmsnorm              - test rmsnorm on GPU");
        eprintln!("  test-rope                 - test rope on GPU");
        eprintln!("  target-forward <gguf> <tok> - run target model forward pass");
        eprintln!("  target-generate <gguf> <start> <n> - cached autoregressive generation");
        eprintln!("  target-info <gguf>        - show target model info");        std::process::exit(1);
    }

    let command = &args[1];

    match command.as_str() {
        "info" => {
            let path = Path::new(&args[2]);
            laguna_raw::run_info(path);
        }
        "shapes" => {
            let path = Path::new(&args[2]);
            laguna_raw::run_shapes(path);
        }
        "forward" => {
            let path = Path::new(&args[2]);
            let token_id: usize = args
                .get(3)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            laguna_raw::run_forward(path, token_id);
        }
        "encode" => {
            let path = Path::new(&args[2]);
            laguna_raw::run_forward_encode(path);
        }
        "test-dequant" => {
            laguna_raw::run_test_dequant();
        }
        "test-matmul" => {
            laguna_raw::run_test_matmul();
        }
        "test-rmsnorm" => {
            laguna_raw::run_test_rmsnorm();
        }
        "test-rope" => {
            laguna_raw::run_test_rope();
        }
        "target-forward" => {
            let path = Path::new(&args[2]);
            let token_id: usize = args
                .get(3)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            laguna_raw::run_target_forward(path, token_id);
        }
        "target-info" => {
            let path = Path::new(&args[2]);
            laguna_raw::run_target_info(path);
        }
        "target-generate" => {
            let path = Path::new(&args[2]);
            let start: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
            let n: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(16);
            laguna_raw::run_target_generate(path, start, n);
        }
        "target-dflash" => {
            let target = Path::new(&args[2]);
            let dflash = Path::new(&args[3]);
            let token_id: usize = args
                .get(4)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            laguna_raw::run_target_dflash(target, dflash, token_id);
        }
        "target-dflash-verify" => {
            let target = Path::new(&args[2]);
            let dflash = Path::new(&args[3]);
            let token_id: usize = args
                .get(4)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            let n_steps: usize = args
                .get(5)
                .and_then(|s| s.parse().ok())
                .unwrap_or(4);
            laguna_raw::run_target_dflash_verify(target, dflash, token_id, n_steps);
        }
        "project-ref" => {
            let path = Path::new(&args[2]);
            let hidden = Path::new(&args[3]);
            laguna_raw::run_project_ref(path, hidden);
        }
        _ => {
            eprintln!("unknown command: {}", command);
            std::process::exit(1);
        }
    }
}
