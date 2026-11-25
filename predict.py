"""
BoliSense Command-Line Inference Utility.
"""

import sys
import argparse
from src.explainability import BoliSenseInferenceEngine


def main():
    parser = argparse.ArgumentParser(description="BoliSense Sentiment Analysis CLI")
    parser.add_argument("--text", type=str, default=None, help="Haryanvi sentence to analyze")
    parser.add_argument("--model", type=str, default="Proposed BoliSense Hybrid", help="Model name")
    parser.add_argument("--compare_all", action="store_true", help="Compare all classical models")
    args = parser.parse_args()

    engine = BoliSenseInferenceEngine()
    if not engine.models:
        print("Error: No models found. Please run 'python train.py' first.")
        sys.exit(1)

    if args.text:
        text = args.text.strip()
        if not text:
            print("Error: Empty sentence provided.")
            sys.exit(1)

        print("=" * 65)
        print("          BoliSense Dialect Sentiment Analysis")
        print("=" * 65)
        print(f"Input:      {text}")

        if args.compare_all:
            results = engine.predict_all(text)
            print("-" * 65)
            print(f"{'Model':<35} | {'Sentiment':<10} | {'Confidence'}")
            print("-" * 65)
            for r in results:
                print(f"{r['Model']:<35} | {r['Sentiment']:<10} | {r['Confidence']}")
            print("-" * 65)
        else:
            res = engine.predict(text, model_name=args.model)
            print(f"Normalized: {res['normalized']}")
            print(f"Model:      {res['model']}")
            print(f"Sentiment:  {res['sentiment']} ({res['confidence']*100:.1f}% confidence)")
            cues = res["explanation"]["cues"]
            neg = cues["negation"]
            con = cues["contrast"]
            lex = cues["lexicon"]
            print("-" * 65)
            print("Linguistic Cues:")
            print(f"  * Negation: {'Detected: ' + str(neg['negation_tokens']) if neg['has_negation'] else 'None'}")
            print(f"  * Contrast: {'Marker: ' + str(con['contrast_marker']) if con['has_contrast'] else 'None'}")
            print(f"  * Lexicon:  Pos={lex['positive_hits']}, Neg={lex['negative_hits']}")
        print("=" * 65)
        sys.exit(0)

    # Interactive Loop
    print("=" * 65)
    print("      BoliSense Interactive Sentiment Analysis Terminal")
    print("      Type a Haryanvi sentence, or 'exit' to quit.")
    print("=" * 65)
    while True:
        try:
            line = input("\nEnter Haryanvi sentence: ").strip()
            if not line or line.lower() in ["exit", "quit", "q"]:
                break
            res = engine.predict(line, model_name=args.model)
            print(f"--> Sentiment:  {res['sentiment']} ({res['confidence']*100:.1f}%)")
            print(f"--> Normalized: {res['normalized']}")
        except (KeyboardInterrupt, EOFError):
            break
    print("Exiting.")


if __name__ == "__main__":
    main()
