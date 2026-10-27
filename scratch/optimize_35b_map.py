import json
from pathlib import Path

map_path = Path("config/atlas_physical_map_35b_3tier.json")
with open(map_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Original 3tier config:", data.get("3tier_configuration"))

# Upgrade all tier2_cold_dma experts to tier1_ram_avx512
# On a 16GB RAM + 8GB VRAM system, all 10,240 experts (14.9 GB) fit entirely in RAM + VRAM!
# Zero disk DMA paging!
upgraded_count = 0
for exp_id, exp_info in data.get("experts", {}).items():
    if exp_info.get("tier") == "tier2_cold_dma":
        exp_info["tier"] = "tier1_ram_avx512"
        upgraded_count += 1

data["3tier_configuration"]["tier0_vram_experts"] = 2000
data["3tier_configuration"]["tier1_ram_experts"] = 8240
data["3tier_configuration"]["tier2_cold_experts"] = 0
data["3tier_configuration"]["direct_io"] = False
data["3tier_configuration"]["elastic_virtual_lock_active"] = True

print(f"Upgraded {upgraded_count} cold experts to RAM-pinned tier1!")
print("New 3tier config:", data.get("3tier_configuration"))

with open(map_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print("Saved optimized config/atlas_physical_map_35b_3tier.json")
