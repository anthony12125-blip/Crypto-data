#!/usr/bin/env python3
"""
Quick Start — Run this first to verify everything works.
No API keys needed (uses public endpoints).
"""

import sys

def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║   🚀 CRYPTO ALPHA BOT — Alt Beta Amplification System            ║
║   Bitcoin is the signal. Altcoins are the trade.                 ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    print("Checking dependencies...")
    missing = []
    for pkg in ["ccxt", "pandas", "numpy", "yaml", "requests"]:
        try:
            __import__(pkg)
            print(f"  ✅ {pkg}")
        except ImportError:
            print(f"  ❌ {pkg} — MISSING")
            missing.append(pkg)

    if missing:
        print(f"\nInstall missing: pip3 install {' '.join(missing)}")
        sys.exit(1)

    print("\n✅ All dependencies OK!\n")

    print("─" * 60)
    print("STEP 1: Scanning BTC Momentum...")
    print("─" * 60)

    from btc_momentum import BTCMomentumScanner
    btc_scanner = BTCMomentumScanner("binance")
    sig = btc_scanner.get_momentum_signal()

    print(f"""
  BTC Price:      ${sig.btc_price:,.2f}
  Momentum Score: {sig.score}/100 [{sig.regime.upper()}]
  1h Change:      {sig.btc_change_1h:+.2f}%
  24h Change:     {sig.btc_change_24h:+.2f}%
  Volume:         {sig.volume_ratio:.1f}x average
  Signal:         {'🟢 GO — Alt trading active' if sig.is_go else '🔴 WAIT — BTC too weak'}
    """)

    print("─" * 60)
    print("STEP 2: Scanning Top 30 Alts for Beta Amplification...")
    print("─" * 60)

    from alt_scanner import AltBetaScanner, print_scan_results
    alt_scanner = AltBetaScanner("binance")
    signals = alt_scanner.scan(btc_momentum_score=sig.score, top_n_alts=30)

    if signals:
        print_scan_results(signals, top_n=15)

    print("""
╔══════════════════════════════════════════════════════════════════╗
║  NEXT STEPS:                                                     ║
║  1. python3 data_collector.py          ← Start NOW               ║
║  2. cp config_example.yaml config.yaml ← Add API keys            ║
║  3. python3 live_trader.py --once      ← Test one cycle           ║
║  4. streamlit run dashboard.py         ← Monitor everything       ║
╚══════════════════════════════════════════════════════════════════╝
    """)

if __name__ == "__main__":
    main()
