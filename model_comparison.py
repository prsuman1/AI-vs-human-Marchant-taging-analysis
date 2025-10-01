#!/usr/bin/env python3
"""
AI Model Comparison Analysis Script
Compares performance of Astra and Qwen models against human-labeled ground truth (DB)

Categories analyzed:
1. Equipment Presence (Boolean)
2. Co-Manufacturing Status (Boolean)
3. Food & Beverage Status (Boolean)
4. Specialty Classification (Multi-class)

Metrics calculated:
- Accuracy
- Precision
- False Positives
- False Negatives
"""

import csv
import json
from collections import defaultdict
from typing import Dict, List, Tuple


class ModelComparator:
    """Compare two AI models against ground truth data"""

    def __init__(self, csv_file_path: str):
        """
        Initialize the comparator with CSV data

        Args:
            csv_file_path: Path to the CSV file containing comparison data
        """
        self.csv_file_path = csv_file_path
        self.data = self._load_data()

        self.categories = {
            'Equipment': ('db_has_equipments', 'astra_has_equipments', 'qwen_has_equipments'),
            'Co-Manufacturing': ('db_is_coman', 'astra_is_coman', 'qwen_is_coman'),
            'Food & Beverage': ('db_is_food_beverage', 'astra_is_food_beverage', 'qwen_is_food_beverage'),
            'Specialty': ('db_specialty', 'astra_specialty', 'qwen_specialty')
        }

    def _load_data(self) -> List[Dict]:
        """Load CSV data into memory"""
        with open(self.csv_file_path, 'r') as f:
            reader = csv.DictReader(f)
            return list(reader)

    def calculate_metrics(self, db_col: str, model_col: str, is_specialty: bool = False) -> Dict:
        """
        Calculate accuracy, precision, and confusion matrix metrics

        Args:
            db_col: Column name for database (ground truth) values
            model_col: Column name for model predictions
            is_specialty: Whether this is the specialty category (exact match)

        Returns:
            Dictionary containing calculated metrics
        """
        if is_specialty:
            # Exact match for specialty
            correct = sum(1 for row in self.data if row[db_col] == row[model_col])
            total = len(self.data)
            accuracy = correct / total * 100
            errors = total - correct

            return {
                'accuracy': accuracy,
                'precision': accuracy,
                'correct': correct,
                'total': total,
                'errors': errors
            }
        else:
            # Boolean fields (True/False or yes/no)
            tp = tn = fp = fn = 0

            for row in self.data:
                db_val = row[db_col].lower() in ['true', 'yes', '1']
                model_val = row[model_col].lower() in ['true', 'yes', '1']

                if db_val and model_val:
                    tp += 1
                elif not db_val and not model_val:
                    tn += 1
                elif not db_val and model_val:
                    fp += 1
                elif db_val and not model_val:
                    fn += 1

            total = len(self.data)
            accuracy = (tp + tn) / total * 100
            precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'tn': tn,
                'fp': fp,
                'fn': fn,
                'total': total
            }

    def analyze_category(self, category: str) -> Tuple[Dict, Dict]:
        """
        Analyze a specific category for both models

        Args:
            category: Category name

        Returns:
            Tuple of (astra_metrics, qwen_metrics)
        """
        db_col, astra_col, qwen_col = self.categories[category]
        is_specialty = (category == 'Specialty')

        astra_metrics = self.calculate_metrics(db_col, astra_col, is_specialty)
        qwen_metrics = self.calculate_metrics(db_col, qwen_col, is_specialty)

        return astra_metrics, qwen_metrics

    def print_category_results(self, category: str, astra_metrics: Dict, qwen_metrics: Dict):
        """Print formatted results for a category"""
        print(f"\n{'─'*80}")
        print(f"CATEGORY: {category.upper()}")
        print(f"{'─'*80}")

        is_specialty = (category == 'Specialty')

        if is_specialty:
            print(f"\nASTRA Performance:")
            print(f"   Accuracy: {astra_metrics['accuracy']:.2f}%")
            print(f"   Correct: {astra_metrics['correct']}/{astra_metrics['total']}")
            print(f"   Errors: {astra_metrics['errors']}")

            print(f"\nQWEN Performance:")
            print(f"   Accuracy: {qwen_metrics['accuracy']:.2f}%")
            print(f"   Correct: {qwen_metrics['correct']}/{qwen_metrics['total']}")
            print(f"   Errors: {qwen_metrics['errors']}")

            if astra_metrics['accuracy'] > qwen_metrics['accuracy']:
                diff = astra_metrics['accuracy'] - qwen_metrics['accuracy']
                print(f"\nWINNER: ASTRA (by {diff:.2f}%)")
            elif qwen_metrics['accuracy'] > astra_metrics['accuracy']:
                diff = qwen_metrics['accuracy'] - astra_metrics['accuracy']
                print(f"\nWINNER: QWEN (by {diff:.2f}%)")
            else:
                print(f"\nTIE")
        else:
            print(f"\nASTRA Performance:")
            print(f"   Accuracy: {astra_metrics['accuracy']:.2f}%")
            print(f"   Precision: {astra_metrics['precision']:.2f}%")
            print(f"   Recall: {astra_metrics['recall']:.2f}%")
            print(f"   F1-Score: {astra_metrics['f1']:.2f}%")
            print(f"   True Positives: {astra_metrics['tp']}")
            print(f"   True Negatives: {astra_metrics['tn']}")
            print(f"   False Positives: {astra_metrics['fp']}")
            print(f"   False Negatives: {astra_metrics['fn']}")

            print(f"\nQWEN Performance:")
            print(f"   Accuracy: {qwen_metrics['accuracy']:.2f}%")
            print(f"   Precision: {qwen_metrics['precision']:.2f}%")
            print(f"   Recall: {qwen_metrics['recall']:.2f}%")
            print(f"   F1-Score: {qwen_metrics['f1']:.2f}%")
            print(f"   True Positives: {qwen_metrics['tp']}")
            print(f"   True Negatives: {qwen_metrics['tn']}")
            print(f"   False Positives: {qwen_metrics['fp']}")
            print(f"   False Negatives: {qwen_metrics['fn']}")

            print(f"\nCOMPARISON:")
            acc_diff = astra_metrics['accuracy'] - qwen_metrics['accuracy']
            prec_diff = astra_metrics['precision'] - qwen_metrics['precision']

            print(f"   Accuracy Difference: {acc_diff:+.2f}% (Astra vs Qwen)")
            print(f"   Precision Difference: {prec_diff:+.2f}% (Astra vs Qwen)")
            print(f"   False Positives: Astra={astra_metrics['fp']}, Qwen={qwen_metrics['fp']}")
            print(f"   False Negatives: Astra={astra_metrics['fn']}, Qwen={qwen_metrics['fn']}")

            if astra_metrics['accuracy'] > qwen_metrics['accuracy']:
                print(f"\nWINNER: ASTRA")
            elif qwen_metrics['accuracy'] > astra_metrics['accuracy']:
                print(f"\nWINNER: QWEN")
            else:
                print(f"\nTIE")

    def calculate_overall_accuracy(self, all_metrics: Dict) -> Tuple[float, float]:
        """
        Calculate overall accuracy across all categories

        Args:
            all_metrics: Dictionary of all calculated metrics

        Returns:
            Tuple of (astra_overall, qwen_overall) accuracy percentages
        """
        astra_total_correct = 0
        qwen_total_correct = 0
        total_predictions = 0

        for category, metrics in all_metrics.items():
            astra_m = metrics['astra']
            qwen_m = metrics['qwen']

            if 'tp' in astra_m:
                astra_total_correct += (astra_m['tp'] + astra_m['tn'])
                qwen_total_correct += (qwen_m['tp'] + qwen_m['tn'])
                total_predictions += astra_m['total']
            else:
                astra_total_correct += astra_m['correct']
                qwen_total_correct += qwen_m['correct']
                total_predictions += astra_m['total']

        astra_overall = (astra_total_correct / total_predictions) * 100
        qwen_overall = (qwen_total_correct / total_predictions) * 100

        return astra_overall, qwen_overall

    def run_full_analysis(self, output_format: str = 'text') -> Dict:
        """
        Run complete analysis on all categories

        Args:
            output_format: 'text' for console output, 'json' for JSON output

        Returns:
            Dictionary containing all results
        """
        if output_format == 'text':
            print("="*80)
            print("COMPARATIVE ANALYSIS: Astra vs Qwen AI Models")
            print("Ground Truth: DB (Human-labeled data)")
            print(f"Total Records: {len(self.data)}")
            print("="*80)
            print("\n" + "="*80)
            print("DETAILED ANALYSIS BY CATEGORY")
            print("="*80)

        all_metrics = {}

        for category in self.categories.keys():
            astra_metrics, qwen_metrics = self.analyze_category(category)
            all_metrics[category] = {
                'astra': astra_metrics,
                'qwen': qwen_metrics
            }

            if output_format == 'text':
                self.print_category_results(category, astra_metrics, qwen_metrics)

        astra_overall, qwen_overall = self.calculate_overall_accuracy(all_metrics)

        if output_format == 'text':
            print("\n" + "="*80)
            print("OVERALL SUMMARY")
            print("="*80)
            print(f"\nASTRA Overall Accuracy: {astra_overall:.2f}%")
            print(f"QWEN Overall Accuracy: {qwen_overall:.2f}%")

            if astra_overall > qwen_overall:
                print(f"\nOVERALL WINNER: ASTRA (by {astra_overall - qwen_overall:.2f}%)")
            elif qwen_overall > astra_overall:
                print(f"\nOVERALL WINNER: QWEN (by {qwen_overall - astra_overall:.2f}%)")
            else:
                print(f"\nOVERALL TIE")

            print("\n" + "="*80)

        # Add overall scores to results
        all_metrics['overall'] = {
            'astra_accuracy': astra_overall,
            'qwen_accuracy': qwen_overall
        }

        if output_format == 'json':
            return all_metrics

        return all_metrics

    def export_results(self, output_file: str, results: Dict):
        """
        Export results to JSON file

        Args:
            output_file: Path to output JSON file
            results: Results dictionary from run_full_analysis
        """
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults exported to: {output_file}")

    def find_disagreements(self, category: str) -> List[Dict]:
        """
        Find cases where Astra and Qwen disagree

        Args:
            category: Category to analyze

        Returns:
            List of disagreement records
        """
        db_col, astra_col, qwen_col = self.categories[category]
        disagreements = []

        for row in self.data:
            if row[astra_col] != row[qwen_col]:
                disagreements.append({
                    'manufacturer_id': row['manufacturer_id'],
                    'db_domain': row['db_domain'],
                    'db_value': row[db_col],
                    'astra_value': row[astra_col],
                    'qwen_value': row[qwen_col]
                })

        return disagreements

    def find_errors(self, category: str, model: str = 'astra') -> List[Dict]:
        """
        Find cases where a model made errors

        Args:
            category: Category to analyze
            model: 'astra' or 'qwen'

        Returns:
            List of error records
        """
        db_col, astra_col, qwen_col = self.categories[category]
        model_col = astra_col if model == 'astra' else qwen_col
        errors = []

        for row in self.data:
            if row[db_col] != row[model_col]:
                errors.append({
                    'manufacturer_id': row['manufacturer_id'],
                    'db_domain': row['db_domain'],
                    'db_value': row[db_col],
                    'model_value': row[model_col]
                })

        return errors


def main():
    """Main entry point"""
    # Initialize comparator
    comparator = ModelComparator('Copy of False positive comparison - no_prompt_combined.csv')

    # Run full analysis
    results = comparator.run_full_analysis(output_format='text')

    # Optionally export to JSON
    # comparator.export_results('analysis_results.json', results)

    # Example: Find disagreements in a specific category
    # disagreements = comparator.find_disagreements('Equipment')
    # print(f"\nFound {len(disagreements)} disagreements in Equipment category")

    # Example: Find errors for a specific model
    # astra_errors = comparator.find_errors('Co-Manufacturing', model='astra')
    # print(f"\nFound {len(astra_errors)} ASTRA errors in Co-Manufacturing category")


if __name__ == '__main__':
    main()