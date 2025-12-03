#!/usr/bin/env python3

import subprocess
import json
import os


class WSLSpotConverter:
    """Uses Spot installed on WSL from Windows"""

    def __init__(self, wsl_script_path="/home/otebook/ltl_to_nbw.py"):
        # Replace 'username' with your actual WSL username
        self.wsl_script_path = wsl_script_path
        self._test_wsl_connection()

    def _test_wsl_connection(self):
        """Test if WSL is accessible"""
        try:
            result = subprocess.run('wsl echo "WSL connected"',
                                    shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print("✓ WSL connection successful")
            else:
                print("✗ WSL connection failed")
        except Exception as e:
            print(f"✗ WSL test failed: {e}")

    def ltl_to_nbw(self, ltl_formula):
        """Call WSL Spot script to convert LTL to NBW"""
        try:
            # Escape the formula for command line
            escaped_formula = ltl_formula.replace('"', '\\"')

            # Build WSL command
            cmd = f'wsl python3 {self.wsl_script_path} "{escaped_formula}"'
            print(f"Executing: {cmd}")

            # Execute command
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                return {'success': False, 'error': result.stderr}

        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Conversion timeout'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def print_automaton_details(self, result, formula):
        """Print automaton details"""
        if not result.get('success', False):
            print(f"❌ Failed to convert: {formula}")
            print(f"   Error: {result.get('error', 'Unknown error')}")
            return

        print(f"\n{'=' * 60}")
        print(f"✅ LTL Formula: {formula}")
        if 'formula_used' in result:
            print(f"   (Converted to: {result['formula_used']})")
        print(f"{'=' * 60}")
        print(f"📊 States: {result['states']}")
        print(f"📊 Edges: {result['edges']}")
        print(f"✅ Acceptance: {result['acceptance']}")
        print(f"🔍 Deterministic: {result['is_deterministic']}")

        # Save HOA to file
        if 'hoa_format' in result:
            safe_name = formula.replace(' ', '_').replace('&', 'and').replace('|', 'or')
            filename = f"automaton_{safe_name}.hoa"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(result['hoa_format'])
            print(f"💾 HOA format saved to: {filename}")

            # Show first few lines of HOA
            lines = result['hoa_format'].split('\n')
            print(f"\n📋 HOA Preview (first 10 lines):")
            for line in lines[:10]:
                print(f"   {line}")


def test_basic_conversion():
    """Test basic LTL to NBW conversion"""
    print("Testing WSL Spot Conversion")
    print("=" * 50)

    # Replace with your actual WSL username in the path below
    converter = WSLSpotConverter("/home/otebook/ltl_to_nbw.py")

    test_formulas = [
        "F p",  # Eventually p
        "G p",  # Always p
        "p U q",  # p until q
        "F G p",  # Infinitely often p
        "p ∧ q",  # p and q (will be converted to p & q)
        "X p",  # Next p
        "¬p",  # Not p (will be converted to !p)
    ]

    for formula in test_formulas:
        print(f"\n🔄 Converting: {formula}")
        result = converter.ltl_to_nbw(formula)
        converter.print_automaton_details(result, formula)


if __name__ == "__main__":
    test_basic_conversion()