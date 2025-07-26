"""
Evaluation Module for Multilingual RAG System
Implements metrics for RAG system evaluation including groundedness, relevance, and quality
"""

import logging
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import re
from datetime import datetime

# Evaluation imports
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from multilingual_rag import MultilingualRAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Structure for evaluation results"""
    query: str
    predicted_answer: str
    ground_truth: str
    retrieved_chunks: List[Dict[str, Any]]
    metrics: Dict[str, float]
    language: str

class RAGEvaluator:
    """
    Comprehensive evaluation system for RAG performance
    """
    
    def __init__(self, rag_system: MultilingualRAG):
        self.rag_system = rag_system
        
        # Initialize embedding model for semantic similarity
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
        
        # Bengali pattern for language detection
        self.bengali_pattern = re.compile(r'[\u0980-\u09FF]+')
        
        logger.info("RAG Evaluator initialized")
    
    def evaluate_groundedness(self, answer: str, retrieved_chunks: List[Dict[str, Any]]) -> float:
        """
        Evaluate if the answer is grounded in the retrieved context
        Returns a score between 0 and 1
        """
        if not retrieved_chunks or not answer.strip():
            return 0.0
        
        try:
            # Combine all retrieved chunks
            context = " ".join([chunk['text'] for chunk in retrieved_chunks])
            
            # Calculate semantic similarity between answer and context
            answer_embedding = self.embedding_model.encode([answer])
            context_embedding = self.embedding_model.encode([context])
            
            similarity = cosine_similarity(answer_embedding, context_embedding)[0][0]
            
            # Also check for lexical overlap
            answer_words = set(answer.lower().split())
            context_words = set(context.lower().split())
            
            if len(answer_words) == 0:
                lexical_overlap = 0.0
            else:
                lexical_overlap = len(answer_words.intersection(context_words)) / len(answer_words)
            
            # Combine semantic and lexical scores
            groundedness_score = (similarity * 0.7) + (lexical_overlap * 0.3)
            
            return min(1.0, max(0.0, groundedness_score))
            
        except Exception as e:
            logger.error(f"Groundedness evaluation failed: {e}")
            return 0.0
    
    def evaluate_relevance(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> float:
        """
        Evaluate if retrieved chunks are relevant to the query
        Returns average relevance score
        """
        if not retrieved_chunks:
            return 0.0
        
        try:
            # Use the similarity scores from retrieval
            similarities = [chunk.get('similarity', 0.0) for chunk in retrieved_chunks]
            
            if not similarities:
                return 0.0
            
            # Return average similarity as relevance score
            relevance_score = np.mean(similarities)
            return min(1.0, max(0.0, relevance_score))
            
        except Exception as e:
            logger.error(f"Relevance evaluation failed: {e}")
            return 0.0
    
    def evaluate_answer_quality(self, predicted: str, ground_truth: str) -> Dict[str, float]:
        """
        Evaluate answer quality using multiple metrics
        """
        metrics = {}
        
        try:
            # Exact match
            metrics['exact_match'] = 1.0 if predicted.strip().lower() == ground_truth.strip().lower() else 0.0
            
            # Semantic similarity
            if predicted.strip() and ground_truth.strip():
                pred_embedding = self.embedding_model.encode([predicted])
                truth_embedding = self.embedding_model.encode([ground_truth])
                semantic_sim = cosine_similarity(pred_embedding, truth_embedding)[0][0]
                metrics['semantic_similarity'] = max(0.0, min(1.0, semantic_sim))
            else:
                metrics['semantic_similarity'] = 0.0
            
            # ROUGE scores
            if predicted.strip() and ground_truth.strip():
                rouge_scores = self.rouge_scorer.score(ground_truth, predicted)
                metrics['rouge1'] = rouge_scores['rouge1'].fmeasure
                metrics['rouge2'] = rouge_scores['rouge2'].fmeasure
                metrics['rougeL'] = rouge_scores['rougeL'].fmeasure
            else:
                metrics['rouge1'] = 0.0
                metrics['rouge2'] = 0.0
                metrics['rougeL'] = 0.0
            
            # BLEU score (approximation for single reference)
            if predicted.strip() and ground_truth.strip():
                # Tokenize for BLEU
                predicted_tokens = predicted.strip().split()
                ground_truth_tokens = [ground_truth.strip().split()]  # List of reference token lists
                
                if predicted_tokens and ground_truth_tokens[0]:
                    bleu_score = sentence_bleu(ground_truth_tokens, predicted_tokens)
                    metrics['bleu'] = bleu_score
                else:
                    metrics['bleu'] = 0.0
            else:
                metrics['bleu'] = 0.0
            
            # Substring match (partial credit)
            if ground_truth.strip().lower() in predicted.strip().lower():
                metrics['substring_match'] = 1.0
            elif predicted.strip().lower() in ground_truth.strip().lower():
                metrics['substring_match'] = 0.8
            else:
                metrics['substring_match'] = 0.0
            
        except Exception as e:
            logger.error(f"Answer quality evaluation failed: {e}")
            # Return default metrics on error
            metrics = {
                'exact_match': 0.0,
                'semantic_similarity': 0.0,
                'rouge1': 0.0,
                'rouge2': 0.0,
                'rougeL': 0.0,
                'bleu': 0.0,
                'substring_match': 0.0
            }
        
        return metrics
    
    def evaluate_single_query(self, query: str, ground_truth: str) -> EvaluationResult:
        """
        Evaluate a single query-answer pair
        """
        try:
            # Get prediction from RAG system
            result = self.rag_system.query(query)
            
            predicted_answer = result['response']
            retrieved_chunks = result['retrieved_chunks']
            language = result['language']
            
            # Calculate metrics
            groundedness = self.evaluate_groundedness(predicted_answer, retrieved_chunks)
            relevance = self.evaluate_relevance(query, retrieved_chunks)
            quality_metrics = self.evaluate_answer_quality(predicted_answer, ground_truth)
            
            # Combine all metrics
            all_metrics = {
                'groundedness': groundedness,
                'relevance': relevance,
                **quality_metrics
            }
            
            return EvaluationResult(
                query=query,
                predicted_answer=predicted_answer,
                ground_truth=ground_truth,
                retrieved_chunks=retrieved_chunks,
                metrics=all_metrics,
                language=language
            )
            
        except Exception as e:
            logger.error(f"Single query evaluation failed: {e}")
            return EvaluationResult(
                query=query,
                predicted_answer="",
                ground_truth=ground_truth,
                retrieved_chunks=[],
                metrics={},
                language="unknown"
            )
    
    def evaluate_test_set(self, test_cases: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Evaluate the RAG system on a test set
        
        Args:
            test_cases: List of dictionaries with 'query' and 'expected_answer' keys
        """
        results = []
        
        logger.info(f"Evaluating {len(test_cases)} test cases...")
        
        for i, test_case in enumerate(test_cases):
            logger.info(f"Evaluating test case {i+1}/{len(test_cases)}")
            
            query = test_case['query']
            expected_answer = test_case['expected_answer']
            
            result = self.evaluate_single_query(query, expected_answer)
            results.append(result)
        
        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics(results)
        
        return {
            'individual_results': [
                {
                    'query': r.query,
                    'predicted_answer': r.predicted_answer,
                    'ground_truth': r.ground_truth,
                    'language': r.language,
                    'metrics': r.metrics,
                    'chunks_retrieved': len(r.retrieved_chunks)
                }
                for r in results
            ],
            'aggregate_metrics': aggregate_metrics,
            'evaluation_summary': self._generate_summary(results, aggregate_metrics)
        }
    
    def _calculate_aggregate_metrics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """Calculate aggregate metrics across all results"""
        if not results:
            return {}
        
        # Collect all metric values
        metric_names = set()
        for result in results:
            metric_names.update(result.metrics.keys())
        
        aggregate = {}
        
        for metric_name in metric_names:
            values = [r.metrics.get(metric_name, 0.0) for r in results if metric_name in r.metrics]
            if values:
                aggregate[f'{metric_name}_mean'] = np.mean(values)
                aggregate[f'{metric_name}_std'] = np.std(values)
                aggregate[f'{metric_name}_min'] = np.min(values)
                aggregate[f'{metric_name}_max'] = np.max(values)
        
        return aggregate
    
    def _generate_summary(self, results: List[EvaluationResult], aggregate_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Generate evaluation summary"""
        
        # Language distribution
        languages = [r.language for r in results]
        language_dist = {lang: languages.count(lang) for lang in set(languages)}
        
        # Performance by language
        perf_by_lang = {}
        for lang in set(languages):
            lang_results = [r for r in results if r.language == lang]
            if lang_results:
                lang_metrics = {}
                for metric in ['groundedness', 'relevance', 'semantic_similarity', 'exact_match']:
                    values = [r.metrics.get(metric, 0.0) for r in lang_results]
                    if values:
                        lang_metrics[metric] = np.mean(values)
                perf_by_lang[lang] = lang_metrics
        
        # Overall performance
        overall_performance = {}
        for metric in ['groundedness', 'relevance', 'semantic_similarity', 'exact_match']:
            values = [r.metrics.get(metric, 0.0) for r in results if metric in r.metrics]
            if values:
                overall_performance[metric] = {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values)
                }
        
        return {
            'total_queries': len(results),
            'language_distribution': language_dist,
            'performance_by_language': perf_by_lang,
            'overall_performance': overall_performance,
            'evaluation_timestamp': datetime.now().isoformat()
        }
    
    def save_evaluation_report(self, evaluation_result: Dict[str, Any], output_path: str):
        """Save evaluation report to JSON file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_result, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"Evaluation report saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save evaluation report: {e}")

# Test cases for the Bengali RAG system
DEFAULT_TEST_CASES = [
    {
        "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
        "expected_answer": "শুম্ভুনাথ"
    },
    {
        "query": "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
        "expected_answer": "মামাকে"
    },
    {
        "query": "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
        "expected_answer": "১৫ বছর"
    },
    {
        "query": "Who is mentioned as a good person according to Anupam?",
        "expected_answer": "Shumbhunath"
    },
    {
        "query": "What was Kalyani's actual age at the time of marriage?",
        "expected_answer": "15 years"
    }
]

def run_evaluation(rag_system: MultilingualRAG, 
                  test_cases: List[Dict[str, str]] = None,
                  output_path: str = "data/evaluation_report.json") -> Dict[str, Any]:
    """
    Run complete evaluation of the RAG system
    """
    if test_cases is None:
        test_cases = DEFAULT_TEST_CASES
    
    evaluator = RAGEvaluator(rag_system)
    results = evaluator.evaluate_test_set(test_cases)
    
    # Save report
    evaluator.save_evaluation_report(results, output_path)
    
    return results

# Example usage
if __name__ == "__main__":
    # Initialize RAG system
    from multilingual_rag import MultilingualRAG
    
    rag = MultilingualRAG()
    
    # Make sure knowledge base is set up
    setup_result = rag.setup_knowledge_base()
    print(f"Setup result: {setup_result['success']}")
    
    if setup_result['success']:
        # Run evaluation
        print("Running evaluation...")
        evaluation_results = run_evaluation(rag)
        
        # Print summary
        summary = evaluation_results['evaluation_summary']
        print(f"\nEvaluation Summary:")
        print(f"Total queries: {summary['total_queries']}")
        print(f"Language distribution: {summary['language_distribution']}")
        
        overall_perf = summary['overall_performance']
        if 'semantic_similarity' in overall_perf:
            print(f"Average semantic similarity: {overall_perf['semantic_similarity']['mean']:.3f}")
        if 'groundedness' in overall_perf:
            print(f"Average groundedness: {overall_perf['groundedness']['mean']:.3f}")
        if 'relevance' in overall_perf:
            print(f"Average relevance: {overall_perf['relevance']['mean']:.3f}")
    else:
        print("Please set up the knowledge base first by adding PDF files to data/pdfs/ directory")
