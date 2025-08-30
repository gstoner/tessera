"""
Tessera-LOTUS Integration for DeepScholar-Bench
Direct implementation following the repository structure and evaluation framework
"""

import tessera as ts
import pandas as pd
from lotus import LOTUS, LM, sem_filter, sem_join, sem_agg, sem_rank
from typing import Dict, List, Any, Optional
import asyncio

class TesseraLotusDeepScholar:
    """
    Integration between Tessera's advanced AI capabilities and LOTUS's 
    semantic operators for DeepScholar-bench evaluation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize Tessera components
        self.tessera_model = self._init_tessera_model()
        
        # Initialize LOTUS with Tessera-optimized LM
        self.lm = TesseraLM(
            model_name=config.get("base_model", "tessera-research-7b"),
            tessera_backend=self.tessera_model
        )
        self.lotus = LOTUS(lm=self.lm)
        
        # DeepScholar-bench evaluation components
        self.evaluator = DeepScholarEvaluator(self.lotus, self.lm)
    
    def _init_tessera_model(self):
        """Initialize Tessera model optimized for research synthesis"""
        return ts.models.ResearchSynthesisTransformer(
            layers=self.config.get("layers", 32),
            dim=self.config.get("dim", 4096),
            heads=self.config.get("heads", 32),
            
            # Advanced features for research synthesis
            hierarchical_reasoning=True,
            flash_mla=True,
            citation_aware_attention=True,
            memory_reduction=0.93,
            
            # DeepScholar-bench specific optimizations
            nugget_extraction=True,
            reference_tracking=True,
            organization_modeling=True
        )
    
    async def research_synthesis_pipeline(
        self,
        query: str,
        arxiv_papers: pd.DataFrame,
        exemplar_related_work: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete research synthesis pipeline following DeepScholar-bench methodology
        
        Args:
            query: Research query (paper topic/scope)
            arxiv_papers: DataFrame with retrieved ArXiv papers
            exemplar_related_work: Human-written exemplar for comparison
        
        Returns:
            Dict containing synthesis result and evaluation metrics
        """
        
        # Stage 1: Semantic filtering of relevant papers
        # Using LOTUS semantic operators with Tessera backend
        relevant_papers = arxiv_papers.sem_filter(
            f"The paper abstract is highly relevant to: {query}",
            lm=self.lm
        )
        
        # Stage 2: Semantic ranking by importance
        ranked_papers = relevant_papers.sem_rank(
            f"Papers most important for understanding {query}",
            top_k=20,
            lm=self.lm
        )
        
        # Stage 3: Nugget extraction with hierarchical reasoning
        nuggets = await self._extract_nuggets(ranked_papers, query)
        
        # Stage 4: Citation-aware synthesis generation
        synthesis_result = await self._generate_synthesis(
            query, ranked_papers, nuggets
        )
        
        # Stage 5: DeepScholar-bench evaluation
        evaluation_metrics = await self.evaluator.evaluate_all_metrics(
            synthesis_result,
            nuggets,
            ranked_papers,
            exemplar_related_work
        )
        
        return {
            "synthesis": synthesis_result,
            "metrics": evaluation_metrics,
            "nuggets": nuggets,
            "relevant_papers": ranked_papers.to_dict('records'),
            "pipeline_info": self._get_pipeline_info()
        }
    
    async def _extract_nuggets(
        self, 
        papers: pd.DataFrame, 
        query: str
    ) -> List[Dict[str, Any]]:
        """Extract information nuggets using hierarchical reasoning"""
        
        nuggets = []
        
        # Use LOTUS semantic aggregation with Tessera's HRM
        for _, paper in papers.iterrows():
            paper_nuggets = await self.lotus.sem_agg(
                pd.DataFrame([paper]),
                f"Extract key information nuggets relevant to {query} from this paper. "
                f"Focus on: methodology, findings, contributions, and limitations.",
                group_by="title",
                lm=self.lm
            )
            
            # Tessera's nugget importance scoring
            for nugget_text in paper_nuggets:
                importance = await self._score_nugget_importance(nugget_text, query)
                nuggets.append({
                    "text": nugget_text,
                    "source": paper["title"],
                    "importance": importance,
                    "arxiv_id": paper.get("arxiv_id", ""),
                })
        
        # Sort by importance and return top nuggets
        return sorted(nuggets, key=lambda x: x["importance"], reverse=True)[:50]
    
    async def _generate_synthesis(
        self,
        query: str,
        papers: pd.DataFrame,
        nuggets: List[Dict[str, Any]]
    ) -> str:
        """Generate research synthesis using Tessera's advanced capabilities"""
        
        # Prepare synthesis context
        context = {
            "query": query,
            "papers": papers.to_dict('records'),
            "nuggets": nuggets,
            "synthesis_strategy": "academic_related_work"
        }
        
        # Use Tessera's hierarchical reasoning for synthesis
        synthesis = await self.tessera_model.research_synthesis(
            context,
            max_length=2048,
            citation_style="academic",
            organization_strategy="logical_flow",
            nugget_coverage_target=0.8
        )
        
        return synthesis
    
    async def _score_nugget_importance(
        self, 
        nugget_text: str, 
        query: str
    ) -> float:
        """Score nugget importance using Tessera's semantic understanding"""
        
        importance_prompt = f"""
        Rate the importance of this information nugget for understanding "{query}":
        Nugget: {nugget_text}
        
        Consider: relevance, novelty, methodological significance, and impact.
        Return a score between 0.0 and 1.0.
        """
        
        score = await self.lm.semantic_score(importance_prompt)
        return float(score)
    
    def _get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline configuration"""
        return {
            "tessera_model": str(self.tessera_model),
            "lotus_version": self.lotus.version,
            "optimization_features": [
                "hierarchical_reasoning", 
                "flash_mla", 
                "citation_aware_attention",
                "93% memory reduction",
                "semantic_operators"
            ],
            "deepscholar_metrics": [
                "organization", "nugget_coverage", "relevance_rate",
                "document_importance", "reference_coverage", 
                "citation_precision", "citation_coverage"
            ]
        }

class TesseraLM:
    """Tessera-optimized Language Model for LOTUS integration"""
    
    def __init__(self, model_name: str, tessera_backend):
        self.model_name = model_name
        self.tessera_backend = tessera_backend
        
    async def complete(self, prompt: str, **kwargs) -> str:
        """Complete text using Tessera backend"""
        return await self.tessera_backend.generate(
            prompt, 
            optimization_mode="research_synthesis",
            **kwargs
        )
    
    async def semantic_score(self, prompt: str) -> float:
        """Semantic scoring using Tessera's advanced attention"""
        return await self.tessera_backend.semantic_score(prompt)

class DeepScholarEvaluator:
    """
    DeepScholar-bench evaluation implementation using LOTUS framework
    Implements all seven evaluation metrics as defined in the benchmark
    """
    
    def __init__(self, lotus: LOTUS, lm: TesseraLM):
        self.lotus = lotus
        self.lm = lm
        
    async def evaluate_all_metrics(
        self,
        synthesis: str,
        nuggets: List[Dict[str, Any]],
        papers: pd.DataFrame,
        exemplar: Optional[str] = None
    ) -> Dict[str, float]:
        """Evaluate all DeepScholar-bench metrics"""
        
        # Knowledge Synthesis metrics
        organization = await self._evaluate_organization(synthesis, exemplar)
        nugget_coverage = await self._evaluate_nugget_coverage(synthesis, nuggets)
        
        # Retrieval Quality metrics
        relevance_rate = await self._evaluate_relevance_rate(synthesis, papers)
        doc_importance = await self._evaluate_document_importance(synthesis, papers)
        ref_coverage = await self._evaluate_reference_coverage(synthesis, papers)
        
        # Verifiability metrics
        citation_precision = await self._evaluate_citation_precision(synthesis, papers)
        citation_coverage = await self._evaluate_citation_coverage(synthesis)
        
        # Composite scores (as per DeepScholar-bench methodology)
        knowledge_synthesis = (organization + nugget_coverage) / 2
        retrieval_quality = (relevance_rate + doc_importance + ref_coverage) / 3
        verifiability = (citation_precision + citation_coverage) / 2
        overall_score = (knowledge_synthesis + retrieval_quality + verifiability) / 3
        
        return {
            # Individual metrics
            "organization": organization,
            "nugget_coverage": nugget_coverage,
            "relevance_rate": relevance_rate,
            "document_importance": doc_importance,
            "reference_coverage": ref_coverage,
            "citation_precision": citation_precision,
            "citation_coverage": citation_coverage,
            
            # Composite scores
            "knowledge_synthesis": knowledge_synthesis,
            "retrieval_quality": retrieval_quality,
            "verifiability": verifiability,
            "overall_score": overall_score
        }
    
    async def _evaluate_organization(
        self, 
        synthesis: str, 
        exemplar: Optional[str]
    ) -> float:
        """Organization: pairwise comparison with human exemplars"""
        
        if not exemplar:
            # Absolute organization assessment
            prompt = f"""
            Evaluate the organization and logical flow of this related work section:
            
            {synthesis}
            
            Rate the organization on a scale of 0.0 to 1.0 considering:
            - Logical flow of ideas
            - Clear transitions between concepts
            - Appropriate section structure
            - Academic writing quality
            """
        else:
            # Pairwise comparison with exemplar
            prompt = f"""
            Compare the organization of these two related work sections:
            
            Exemplar (human-written):
            {exemplar}
            
            Generated:
            {synthesis}
            
            Rate how well the generated section's organization compares to the exemplar (0.0 to 1.0):
            """
        
        score = await self.lm.semantic_score(prompt)
        return float(score)
    
    async def _evaluate_nugget_coverage(
        self, 
        synthesis: str, 
        nuggets: List[Dict[str, Any]]
    ) -> float:
        """Nugget Coverage: efficiency in capturing key information"""
        
        covered_nuggets = 0
        total_nuggets = len(nuggets)
        
        for nugget in nuggets:
            coverage_prompt = f"""
            Does this related work section adequately cover the following information:
            Nugget: {nugget['text']}
            
            Related work section:
            {synthesis}
            
            Answer: Yes (1) or No (0)
            """
            
            is_covered = await self.lm.semantic_score(coverage_prompt)
            if is_covered > 0.5:
                covered_nuggets += 1
        
        return covered_nuggets / total_nuggets if total_nuggets > 0 else 0.0
    
    async def _evaluate_relevance_rate(
        self, 
        synthesis: str, 
        papers: pd.DataFrame
    ) -> float:
        """Relevance Rate: relevance of retrieved sources"""
        
        relevance_scores = []
        
        for _, paper in papers.iterrows():
            relevance_prompt = f"""
            How relevant is this paper to the research synthesis:
            
            Paper: {paper['title']} - {paper.get('abstract', 'No abstract')}
            
            Synthesis: {synthesis}
            
            Rate relevance from 0.0 to 1.0:
            """
            
            relevance = await self.lm.semantic_score(relevance_prompt)
            relevance_scores.append(float(relevance))
        
        return sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
    
    async def _evaluate_citation_precision(
        self, 
        synthesis: str, 
        papers: pd.DataFrame
    ) -> float:
        """Citation Precision: whether each citation supports the given claim"""
        
        # Extract citations from synthesis
        citations = self._extract_citations(synthesis)
        
        if not citations:
            return 0.0
        
        precision_scores = []
        
        for citation in citations:
            # Find the source paper
            source_paper = self._find_source_paper(citation, papers)
            
            if source_paper is not None:
                support_prompt = f"""
                Does this citation accurately represent the source paper?
                
                Citation: {citation['text']}
                Source: {source_paper.get('title', '')} - {source_paper.get('abstract', '')}
                
                Rate accuracy from 0.0 to 1.0:
                """
                
                support_score = await self.lm.semantic_score(support_prompt)
                precision_scores.append(float(support_score))
            else:
                precision_scores.append(0.0)  # Citation not found in sources
        
        return sum(precision_scores) / len(precision_scores)
    
    def _extract_citations(self, synthesis: str) -> List[Dict[str, Any]]:
        """Extract citations from synthesis text"""
        # Simplified citation extraction - in practice would use more sophisticated parsing
        import re
        
        citations = []
        # Look for common citation patterns
        citation_patterns = [
            r'\([^)]*\d{4}[^)]*\)',  # (Author, 2024)
            r'\[[^\]]+\]',           # [1]
        ]
        
        for pattern in citation_patterns:
            matches = re.findall(pattern, synthesis)
            for match in matches:
                citations.append({
                    "text": match,
                    "context": synthesis  # In practice, extract surrounding context
                })
        
        return citations
    
    def _find_source_paper(
        self, 
        citation: Dict[str, Any], 
        papers: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """Find the source paper for a citation"""
        # Simplified matching - in practice would use more sophisticated matching
        citation_text = citation["text"].lower()
        
        for _, paper in papers.iterrows():
            title = paper.get("title", "").lower()
            if any(word in citation_text for word in title.split() if len(word) > 3):
                return paper.to_dict()
        
        return None
    
    async def _evaluate_document_importance(
        self, 
        synthesis: str, 
        papers: pd.DataFrame
    ) -> float:
        """Document Importance: citation counts of references"""
        
        # In practice, would look up actual citation counts
        # For demonstration, using paper importance assessment
        importance_scores = []
        
        for _, paper in papers.iterrows():
            importance_prompt = f"""
            Rate the importance of this paper in the field (based on potential citations):
            
            Paper: {paper['title']}
            Abstract: {paper.get('abstract', 'No abstract')}
            
            Rate from 0.0 to 1.0:
            """
            
            importance = await self.lm.semantic_score(importance_prompt)