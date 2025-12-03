#!/usr/bin/env python3

import sys
import os

# Add the path to your main module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ltllim_parser import EnhancedLTLimProcessor, QuantitativeKripkeStructure


def run_enhanced_test_suite():
    """Run the enhanced test suite focusing on MSCC detection and LP model checking"""

    # Create a comprehensive Quantitative Kripke Structure for testing
    qks = QuantitativeKripkeStructure(
        states={'s0', 's1', 's2', 's3'},
        init_state='s0',
        edges={
            ('s0', 's1'), ('s0', 's2'),
            ('s1', 's0'), ('s1', 's2'), ('s1', 's3'),
            ('s2', 's1'), ('s2', 's3'),
            ('s3', 's0'), ('s3', 's2')
        },
        boolean_vars={'p', 'q', 'r'},
        logical_formulas={
            's0': {'p'},
            's1': {'q'},
            's2': {'p', 'r'},
            's3': {'q', 'r'}
        },
        numeric_values={
            's0': {'x': 1.0, 'y': 2.0, 'z': 0.5},
            's1': {'x': 3.0, 'y': 1.0, 'z': 1.5},
            's2': {'x': 2.0, 'y': 3.0, 'z': 0.8},
            's3': {'x': 4.0, 'y': 0.5, 'z': 2.0}
        }
    )

    # Use the enhanced processor with the QKS
    processor = EnhancedLTLimProcessor("/home/otebook/ltl_to_nbw.py", qks)

    # =============================================================================
    # ENHANCED TEST CATEGORIES FOCUSING ON NEW FEATURES
    # =============================================================================

    test_categories = {
        "MSCC DETECTION & FAIRNESS TESTS": [
            "G p",  # Should find fair MSCCs with p
            "F q",  # Should find fair MSCCs with q
            "G r",  # Should find fair MSCCs with r
            "p U q",  # Complex temporal with fairness
        ],

        "LIMIT-AVERAGE LP FEASIBILITY TESTS": [
            "LimInfAvg(x) >= 2.0",  # Should be feasible (x values: 1,3,2,4)
            "LimInfAvg(x) <= 3.0",  # Should be feasible
            "LimInfAvg(y) >= 1.0",  # Should be feasible (y values: 2,1,3,0.5)
            "LimInfAvg(y) <= 2.5",  # Should be feasible
            "LimInfAvg(z) > 0.0",  # Should be feasible (all z > 0)
        ],

        "LIMIT-AVERAGE LP INFEASIBILITY TESTS": [
            "LimInfAvg(x) >= 5.0",  # Should be infeasible (max x=4)
            "LimInfAvg(x) <= 0.5",  # Should be infeasible (min x=1)
            "LimInfAvg(y) >= 4.0",  # Should be infeasible (max y=3)
            "LimInfAvg(z) < 0.0",  # Should be infeasible (all z >= 0.5)
        ],

        "COMPLEX LIMIT-AVERAGE COMBINATIONS": [
            "LimInfAvg(x) >= 2.0 ∧ LimSupAvg(y) <= 3.0",  # Both feasible
            "LimInfAvg(x) >= 1.0 ∧ LimInfAvg(y) >= 1.0",  # Both feasible
            "LimInfAvg(x) >= 3.0 ∧ LimInfAvg(y) >= 2.0",  # May be feasible
        ],

        "TEMPORAL + LIMIT-AVERAGE INTEGRATION": [
            "F(LimInfAvg(x) >= 2.0)",  # Eventually satisfy limit-average
            "G(LimInfAvg(y) <= 3.0)",  # Always satisfy limit-average
            "p U (LimInfAvg(z) > 1.0)",  # Until limit-average satisfied
            "F(LimInfAvg(x) >= 2.0 ∧ q)",  # Combination with boolean
        ]
    }

    # =============================================================================
    # TEST EXECUTION WITH ENHANCED MONITORING
    # =============================================================================

    print(" ENHANCED LTLim PROCESSOR TEST SUITE - MSCC & LP FEATURES")
    print("=" * 100)

    total_tests = 0
    passed_tests = 0
    failed_tests = 0

    for category, formulas in test_categories.items():
        print(f"\n{'#' * 100}")
        print(f"📂 TEST CATEGORY: {category}")
        print(f"{'#' * 100}")

        for i, formula in enumerate(formulas, 1):
            total_tests += 1
            print(f"\n🔬 Test {total_tests}: {formula}")
            print("-" * 80)

            try:
                # Reset processor state for clean test
                processor.variables.clear()
                processor.propositions.clear()
                processor.limit_avg_assertions.clear()

                # Parse the formula first to check syntax
                parse_tree = processor.parse(formula)
                parsed_str = processor.tree_to_string(parse_tree)
                print(f"   ✅ Parsed successfully: {parsed_str}")

                # Test the complete pipeline with product construction
                print(f"   🚀 Starting enhanced pipeline with MSCC detection and LP checking...")
                results = processor.complete_pipeline_with_product(formula)

                if results:
                    successful_products = sum(1 for _, _, product in results if product is not None)
                    print(f"   ✅ Pipeline completed: {successful_products}/{len(results)} successful products")

                    # Enhanced analysis of results
                    for j, (chi, xi, product) in enumerate(results):
                        print(f"\n      📊 Disjunct {j + 1} Analysis:")
                        print(f"        χ: {chi}")
                        print(f"        ξ: {xi}")

                        if product:
                            print(f"        ✅ Product built successfully")
                            print(
                                f"        📈 Product stats: {len(product.states)} states, {len(product.accepting_states)} accepting")

                            # Check if we found fair computations
                            has_fair = processor._check_fair_computations(product, chi)
                            print(f"        🎯 Fair computations with χ: {has_fair}")

                            if has_fair:
                                passed_tests += 1
                                print(f"        ✅ TEST PASSED - Fair computations found satisfying χ")
                            else:
                                failed_tests += 1
                                print(f"        ❌ TEST FAILED - No fair computations found")
                        else:
                            print(f"        ❌ Product construction failed")
                            failed_tests += 1

                else:
                    print(f"   ❌ Pipeline failed - no results generated")
                    failed_tests += 1

            except SyntaxError as e:
                print(f"   ❌ Syntax error: {e}")
                failed_tests += 1
            except Exception as e:
                print(f"   💥 Unexpected error: {e}")
                import traceback
                traceback.print_exc()
                failed_tests += 1

            print("-" * 80)

    # =============================================================================
    # FOCUSED MSCC DETECTION TESTS
    # =============================================================================

    print(f"\n{'=' * 100}")
    print("🔍 FOCUSED MSCC DETECTION TESTS")
    print(f"{'=' * 100}")

    mscc_test_formulas = [
        "G p",  # Simple case - should find MSCCs
        "F q",  # Another simple case
    ]

    for formula in mscc_test_formulas:
        try:
            print(f"\n🧪 Testing MSCC detection for: {formula}")
            processor.variables.clear()
            processor.propositions.clear()

            # Run partial pipeline to get NBW results
            nbw_results = processor.complete_pipeline_with_nbw(formula)

            if nbw_results:
                for chi, xi, nbw_result in nbw_results:
                    if nbw_result.get('success', False):
                        print(f"   Building product for MSCC analysis...")
                        product = processor.build_product_for_disjunct(chi, xi, nbw_result)
                        if product:
                            # Test MSCC detection directly
                            print(f"   🔍 Running MSCC detection...")
                            fair_msccs = processor.find_fair_msccs(product)
                            print(f"   📊 Found {len(fair_msccs)} fair MSCCs")

                            for i, mscc in enumerate(fair_msccs):
                                accepting_count = sum(1 for state in mscc if state in product.accepting_states)
                                print(f"      MSCC {i + 1}: {len(mscc)} states, {accepting_count} accepting")
        except Exception as e:
            print(f"   ❌ MSCC test failed: {e}")

    # =============================================================================
    # LP CONSTRAINT PARSING TESTS
    # =============================================================================

    print(f"\n{'=' * 100}")
    print("🧮 LP CONSTRAINT PARSING TESTS")
    print(f"{'=' * 100}")

    constraint_tests = [
        "LimInfAvg(x) >= 2.0",
        "LimSupAvg(y) <= 3.0",
        "LimInfAvg(z) > 1.0",
        "LimInfAvg(x) >= 2.0 ∧ LimSupAvg(y) <= 3.0",
        "¬(LimInfAvg(x) >= 5.0)",
    ]

    for test in constraint_tests:
        try:
            print(f"\n📐 Testing constraint parsing: {test}")
            constraints = processor._parse_limit_avg_formula(test)
            print(f"   ✅ Parsed {len(constraints)} constraints")
            for constraint in constraints:
                print(f"      {constraint}")
        except Exception as e:
            print(f"   ❌ Constraint parsing failed: {e}")

    # =============================================================================
    # TEST SUMMARY
    # =============================================================================

    print(f"\n{'=' * 100}")
    print("📊 ENHANCED TEST SUMMARY")
    print(f"{'=' * 100}")
    print(f"Total tests: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {failed_tests}")
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    print(f"📈 Success rate: {success_rate:.1f}%")

    # Detailed feature report
    print(f"\n🎯 FEATURE VERIFICATION REPORT:")
    print(f"   ✓ MSCC Detection: {'Implemented' if hasattr(processor, '_find_all_msccs') else 'Missing'}")
    print(f"   ✓ Fair MSCC Filtering: {'Implemented' if hasattr(processor, 'find_fair_msccs') else 'Missing'}")
    print(f"   ✓ LP-based ComponentCheck: {'Implemented' if hasattr(processor, 'component_check') else 'Missing'}")
    print(
        f"   ✓ Geometric Constraint Parsing: {'Implemented' if hasattr(processor, '_parse_limit_avg_formula') else 'Missing'}")

    print(f"\n🎉 ENHANCED TESTING COMPLETE!")

    return passed_tests, failed_tests


if __name__ == "__main__":
    # Run the enhanced test suite
    passed, failed = run_enhanced_test_suite()

    # Exit with appropriate code for CI/CD
    if failed > 0:
        print(f"\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)
    else:
        print(f"\n✅ All tests passed! Your MSCC detection and LP model checking are working correctly.")
        sys.exit(0)