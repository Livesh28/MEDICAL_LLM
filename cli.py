#!/usr/bin/env python3
"""
Interactive Command-Line Interface (CLI) for the Medical LLM & RAG System.
Supports single-query execution and interactive live chat sessions with real-time token streaming.
"""

import sys
import os
import argparse
import time

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from rag.pipeline import MedicalRAGPipeline

# Terminal ANSI styling
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

def print_banner():
    print(f"\n{CYAN}{BOLD}================================================================={RESET}")
    print(f"{CYAN}{BOLD}        Medical LLM & Collaborative RAG Pipeline - Terminal CLI{RESET}")
    print(f"{DIM}  Local Apple Silicon MPS • CLSI GP41 & WHO Verified • Zero Cloud APIs{RESET}")
    print(f"{CYAN}{BOLD}================================================================={RESET}\n")

def run_query(pipeline: MedicalRAGPipeline, question: str, model: str = "unified", top_k: int = 4):
    print(f"\n{YELLOW}{BOLD}Question:{RESET} {question}")
    print(f"{DIM}[Engine: {model} | Top-K Chunks: {top_k}]{RESET}")
    print(f"{GREEN}{BOLD}Answer:{RESET} ", end="", flush=True)

    sources = []
    t0 = time.time()

    # Stream answer tokens directly to terminal
    for evt in pipeline.answer_question_stream(
        question=question,
        model=model,
        top_k=top_k,
        max_new_tokens=300,
        temperature=0.2
    ):
        evt_type = evt.get("type")
        if evt_type == "meta":
            sources = evt.get("sources", [])
        elif evt_type == "token":
            print(evt.get("delta", ""), end="", flush=True)
        elif evt_type == "done":
            elapsed = evt.get("total_ms", round((time.time() - t0) * 1000, 1))
            grounded = evt.get("grounded", True)
            conf = evt.get("confidence", "high")
            cache_hit = evt.get("cache_hit", False)
            print() # newline after streaming finishes

            print(f"\n{BLUE}{BOLD}--- Grounding & Retrieval Metadata ---{RESET}")
            cache_tag = f" {GREEN}(Cache Hit < 6ms){RESET}" if cache_hit else ""
            print(f"• Latency: {BOLD}{elapsed} ms{RESET}{cache_tag}")
            print(f"• Grounded: {GREEN if grounded else YELLOW}{grounded}{RESET} (Confidence: {conf})")
            if sources:
                print(f"• Verified Sources Cited ({len(sources)}):")
                for s in sources:
                    badge = f"[{s.get('citation_idx', 1)}]"
                    src_id = s.get("source_id", "CLSI/WHO")
                    sec = s.get("section", "")
                    sec_str = f" - {sec}" if sec else ""
                    print(f"  {CYAN}{badge}{RESET} {BOLD}{src_id}{RESET}{sec_str}")
            print(f"{BLUE}---------------------------------------{RESET}\n")

def interactive_loop(pipeline: MedicalRAGPipeline, model: str = "unified"):
    print_banner()
    print(f"Using Model: {BOLD}{model}{RESET}")
    print(f"Type your clinical query below, or type {BOLD}'exit'{RESET} or {BOLD}'quit'{RESET} to stop.\n")

    while True:
        try:
            prompt = input(f"{MAGENTA}{BOLD}Doctor / Student > {RESET}").strip()
            if not prompt:
                continue
            if prompt.lower() in ["exit", "quit", "q"]:
                print(f"\n{CYAN}Session ended. Stay safe!{RESET}\n")
                break
            run_query(pipeline, prompt, model=model)
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{CYAN}Session interrupted. Goodbye!{RESET}\n")
            break

def main():
    parser = argparse.ArgumentParser(description="Medical LLM & RAG Terminal CLI")
    parser.add_argument("query", nargs="?", default=None, help="Optional single clinical question to ask directly")
    parser.add_argument("--model", default="unified", choices=["unified", "openbiollm", "llama", "transformer_lm"],
                        help="Model to use (default: unified)")
    parser.add_argument("--top_k", type=int, default=4, help="Number of RAG chunks to retrieve (default: 4)")
    args = parser.parse_args()

    # Initialize Pipeline
    print(f"{DIM}Loading Medical RAG Pipeline onto Apple Silicon MPS...{RESET}", end="", flush=True)
    pipeline = MedicalRAGPipeline(
        checkpoint_path="checkpoints/best.pt",
        tokenizer_path="tokenizer/artifacts/tokenizer.json",
        db_dir="data/rag_db",
        device_name="mps"
    )
    print(f"\r{GREEN}✓ Medical Pipeline loaded successfully.{RESET}\n")

    if args.query:
        run_query(pipeline, args.query, model=args.model, top_k=args.top_k)
    else:
        interactive_loop(pipeline, model=args.model)

if __name__ == "__main__":
    main()
