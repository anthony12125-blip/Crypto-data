# Alternative: Train on Mac Mini

If cloud deployment fails, you can train locally on your Mac Mini.

**Command:**
```bash
python3 train_local.py
```

**Trade-offs:**
- **Slower**: Mac MPS vs GPU
- **Sequential**: Trains one agent at a time  
- **Cheaper**: No cloud costs
- **Overnight**: Set it and forget it

**Requirements:**
- 16GB+ RAM recommended
- M1/M2 Mac for MPS acceleration
- ~50GB free disk space

Models save to: `~/iron_dragoon_training/models/`
