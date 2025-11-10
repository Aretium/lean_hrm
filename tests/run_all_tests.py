"""
Master test runner for all layer tests

Runs all test scripts in sequence and reports results.
"""

import sys
import subprocess


def run_test(test_name, test_file):
    """Run a single test file and return success status."""
    print("\n" + "="*80)
    print(f"Running: {test_name}")
    print("="*80)

    try:
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=False,
            text=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError:
        print(f"\n❌ {test_name} FAILED!")
        return False


def main():
    """Run all tests."""
    tests = [
        ("Basic Layers (Linear, RMSNorm)", "test_basic_layers.py"),
        ("Rotary Position Embeddings (RoPE)", "test_rope.py"),
        ("SwiGLU Activation", "test_swiglu.py"),
        ("Multi-Head Attention", "test_attention.py"),
        ("Transformer Block", "test_transformer_block.py"),
    ]

    print("\n" + "="*80)
    print("MLX-HRM LAYER TEST SUITE")
    print("="*80)
    print(f"Running {len(tests)} test suites...\n")

    results = {}
    for test_name, test_file in tests:
        results[test_name] = run_test(test_name, test_file)

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = 0
    failed = 0

    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status:12s} - {test_name}")
        if success:
            passed += 1
        else:
            failed += 1

    print("="*80)
    print(f"Total: {passed}/{len(tests)} test suites passed")

    if failed == 0:
        print("\n🎉 ALL TEST SUITES PASSED! 🎉\n")
        return 0
    else:
        print(f"\n⚠️  {failed} test suite(s) failed\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
