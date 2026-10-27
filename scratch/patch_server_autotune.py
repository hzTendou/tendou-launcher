# scratch/patch_server_autotune.py

with open('src/atlas/server.py', 'rb') as f:
    content = f.read().decode('utf-8')

# 1. Update candidates list to include tendou-v0.13.0-stable
old_cand = """    candidates = [
        os.path.join(base_dir, "llama.cpp", "build", "bin", "Release", "tendou-engine.exe"),
        os.path.join(base_dir, "llama.cpp", "build", "bin", "Release", "llama-atlas-engine.exe"),
        os.path.join(base_dir, "releases", "tendou-v0.11.0-stable", "bin", "tendou-engine.exe"),
        os.path.join(base_dir, "releases", "tendou-v0.9.0-stable", "bin", "tendou-engine.exe"),
    ]"""

new_cand = """    candidates = [
        os.path.join(base_dir, "releases", "tendou-v0.13.0-stable", "bin", "tendou-engine.exe"),
        os.path.join(base_dir, "llama.cpp", "build", "bin", "Release", "tendou-engine.exe"),
        os.path.join(base_dir, "llama.cpp", "build", "bin", "Release", "llama-atlas-engine.exe"),
        os.path.join(base_dir, "releases", "tendou-v0.11.0-stable", "bin", "tendou-engine.exe"),
        os.path.join(base_dir, "releases", "tendou-v0.9.0-stable", "bin", "tendou-engine.exe"),
    ]"""

assert old_cand.replace('\r\n', '\n') in content.replace('\r\n', '\n'), "old_cand not found"
content = content.replace(old_cand.replace('\n', '\r\n'), new_cand.replace('\n', '\r\n'))

# 2. Update physical map check for 35B to check 3tier map first
old_phys = """            elif self.model_path and ("35B" in self.model_path or "35b" in self.model_path):
                cand = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "config", "atlas_physical_map_35b.json"
                )
                if os.path.exists(cand):
                    cmd.extend(["--atlas-physical-map", cand])"""

new_phys = """            elif self.model_path and ("35B" in self.model_path or "35b" in self.model_path):
                cand_3tier = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "config", "atlas_physical_map_35b_3tier.json"
                )
                cand = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "config", "atlas_physical_map_35b.json"
                )
                if os.path.exists(cand_3tier):
                    cmd.extend(["--atlas-physical-map", cand_3tier])
                elif os.path.exists(cand):
                    cmd.extend(["--atlas-physical-map", cand])"""

assert old_phys.replace('\r\n', '\n') in content.replace('\r\n', '\n'), "old_phys not found"
content = content.replace(old_phys.replace('\n', '\r\n'), new_phys.replace('\n', '\r\n'))

# 3. Add --autotune and --profile options to create_parser()
old_parser_tail = """    parser.add_argument("--tlep-bw", type=int, default=128,
                        help="TLEP predictive prefetch bandwidth budget in MB (default: 128)")
    return parser"""

new_parser_tail = """    parser.add_argument("--tlep-bw", type=int, default=128,
                        help="TLEP predictive prefetch bandwidth budget in MB (default: 128)")
    parser.add_argument("--autotune", action="store_true",
                        help="Enable Universal Auto-Tuning Engine: automatically inspects hardware and model topology to set optimal parameters for maximum TPS")
    parser.add_argument("--profile", choices=["auto", "manual"], default="manual",
                        help="Configuration profile mode (default: manual; use 'auto' for automated hardware-aware tuning)")
    return parser"""

assert old_parser_tail.replace('\r\n', '\n') in content.replace('\r\n', '\n'), "old_parser_tail not found"
content = content.replace(old_parser_tail.replace('\n', '\r\n'), new_parser_tail.replace('\n', '\r\n'))

# 4. Integrate autotune logic in main()
old_main_top = """def main(argv=None):
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.aggressive_prefetch:"""

new_main_top = """def main(argv=None):
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.autotune or args.profile == "auto":
        try:
            from src.atlas.autotune import probe_hardware, inspect_model_topology, solve_optimal_configuration
            hw = probe_hardware()
            if os.path.exists(args.model_path):
                model_topo = inspect_model_topology(args.model_path)
                auto_cfg = solve_optimal_configuration(hw, model_topo)
                logger.info(
                    f"[AUTOTUNE] Solved optimal profile for {model_topo.model_name}: "
                    f"threads={auto_cfg.threads}, ngl={auto_cfg.gpu_layers}, "
                    f"vram_cache={auto_cfg.vram_cache_mb}MB, moe_cache={auto_cfg.moe_cache_mib}MB, "
                    f"mtp={auto_cfg.mtp_mode} (draft_n={auto_cfg.mtp_draft_n})"
                )
                args.threads = auto_cfg.threads
                args.ngl = auto_cfg.gpu_layers
                args.vram_cache = auto_cfg.vram_cache_mb
                args.moe_cache_mib = auto_cfg.moe_cache_mib
                args.mtp = auto_cfg.mtp_mode
                args.mtp_draft_n = auto_cfg.mtp_draft_n
                args.boost_level = auto_cfg.boost_level
        except Exception as e:
            logger.warning(f"[AUTOTUNE] Auto-tuning resolution encountered warning: {e}. Falling back to manual parameters.")

    if args.aggressive_prefetch:"""

assert old_main_top.replace('\r\n', '\n') in content.replace('\r\n', '\n'), "old_main_top not found"
content = content.replace(old_main_top.replace('\n', '\r\n'), new_main_top.replace('\n', '\r\n'))

with open('src/atlas/server.py', 'wb') as f:
    f.write(content.encode('utf-8'))
print("Updated src/atlas/server.py successfully.")
