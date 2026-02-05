import streamlit as st
import os
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Tuple
import docx
import json
from datetime import datetime
import re
import tempfile
import shutil

class QADocument:
    """Represents a Q&A document"""
    def __init__(self, filename, full_text, file_path, file_modified_date, word_count, qa_pairs=None):
        self.filename = filename
        self.full_text = full_text
        self.file_path = file_path
        self.file_modified_date = file_modified_date
        self.word_count = word_count
        self.qa_pairs = qa_pairs if qa_pairs is not None else []

class PharmaQASearcher:
    def __init__(self, model_name: str = 'pritamdeka/S-PubMedBert-MS-MARCO'):
        """Initialize the Q&A document searcher with a medical model"""
        st.info(f"Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.index = None
        self.documents: List[QADocument] = []
        
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract all text from a Word document"""
        try:
            doc = docx.Document(file_path)
            text_parts = []
            
            # Get paragraph text
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_parts.append(paragraph.text)
            
            # Get table text
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        if cell.text.strip():
                            text_parts.append(cell.text)
            
            return '\n'.join(text_parts)
        except Exception as e:
            st.error(f"Error reading {file_path}: {str(e)}")
            return ""
    
    def extract_qa_pairs(self, text: str) -> List[Tuple[str, str]]:
        """
        Attempt to extract Q&A pairs from text
        This is a simple heuristic - adjust based on your document format
        """
        qa_pairs = []
        lines = text.split('\n')
        
        current_question = None
        current_answer = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line looks like a question
            if (line.endswith('?') or 
                line.lower().startswith('q:') or 
                line.lower().startswith('question:')):
                # Save previous Q&A pair if exists
                if current_question and current_answer:
                    qa_pairs.append((current_question, '\n'.join(current_answer)))
                
                # Start new question
                current_question = line
                current_answer = []
            elif current_question:
                # Add to current answer
                current_answer.append(line)
        
        # Add last Q&A pair
        if current_question and current_answer:
            qa_pairs.append((current_question, '\n'.join(current_answer)))
        
        return qa_pairs
    
    def get_sentence_similarity(self, sentence: str, query: str) -> float:
        """
        Calculate cosine similarity between a sentence and query using the model
        """
        try:
            # Encode both texts
            sentence_emb = self.model.encode([sentence], convert_to_numpy=True, show_progress_bar=False)
            query_emb = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False)
            
            # Normalize
            faiss.normalize_L2(sentence_emb)
            faiss.normalize_L2(query_emb)
            
            # Calculate cosine similarity (dot product of normalized vectors)
            similarity = np.dot(sentence_emb[0], query_emb[0])
            
            return float(similarity)
        except:
            return 0.0
    
    def highlight_text(self, text: str, query: str, max_length: int = 1500) -> str:
        """
        Highlight query terms in text - enhanced for medical terminology
        """
        if not query or not text:
            return text[:max_length] + ("..." if len(text) > max_length else "")
        
        # Extract query terms including medical/drug names
        stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by', 'this', 'that', 'these', 'those', 'are', 'was', 'were', 'be', 'been', 'being', 'can', 'could', 'should', 'would', 'will', 'what', 'when', 'where', 'who', 'how', 'does', 'do', 'did', 'has', 'have', 'had'}
        
        query_words = query.split()
        query_terms = []
        
        for word in query_words:
            word_clean = word.lower().strip('?.,!;:')
            if len(word_clean) > 2 and word_clean not in stop_words:
                query_terms.append(word_clean)
            # Preserve case for drug names
            if word[0].isupper() and len(word) > 3:
                query_terms.append(word.strip('?.,!;:'))
        
        query_terms = list(dict.fromkeys(query_terms))
        
        if not query_terms:
            return text[:max_length] + ("..." if len(text) > max_length else "")
        
        # Find first occurrence of any query term
        text_lower = text.lower()
        first_match = len(text)
        
        for term in query_terms:
            pos = text_lower.find(term.lower())
            if pos != -1 and pos < first_match:
                first_match = pos
        
        if first_match == len(text):
            excerpt = text[:max_length]
        else:
            start = max(0, first_match - 200)
            end = min(len(text), start + max_length)
            excerpt = text[start:end]
        
        # Highlight with variations
        highlighted = excerpt
        sorted_terms = sorted(query_terms, key=len, reverse=True)
        
        for term in sorted_terms:
            # Multiple patterns for better matching
            patterns = [
                # Exact match
                re.compile(r'\b(' + re.escape(term) + r')\b', re.IGNORECASE),
                # Word starting with term
                re.compile(r'\b(' + re.escape(term.lower()) + r'\w*)\b', re.IGNORECASE) if len(term) > 4 else None,
            ]
            
            for pattern in patterns:
                if pattern:
                    highlighted = pattern.sub(
                        lambda m: f'<mark style="background-color: #ffeb3b; padding: 2px 4px; border-radius: 3px; font-weight: 500;">{m.group(1)}</mark>',
                        highlighted
                    )
        
        prefix = "..." if first_match > 200 else ""
        suffix = "..." if end < len(text) else ""
        
        return prefix + highlighted + suffix
    
    def get_relevant_snippets(self, text: str, query: str, num_snippets: int = 3, snippet_length: int = 500) -> List[str]:
        """
        Extract multiple relevant snippets from text based on semantic similarity
        Enhanced highlighting for medical terminology with proper HTML handling
        """
        if not query or not text:
            return []
        
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not sentences:
            return []
        
        # Calculate similarity for each sentence (this is fast)
        sentence_similarities = []
        
        for i, sentence in enumerate(sentences):
            similarity = self.get_sentence_similarity(sentence, query)
            if similarity > 0.25:  # Lower threshold to get more candidates
                sentence_similarities.append((similarity, i, sentence))
        
        if not sentence_similarities:
            return []
        
        # Sort by similarity
        sentence_similarities.sort(reverse=True)
        
        # Get top snippets
        snippets = []
        used_positions = set()
        
        # Enhanced keyword extraction - include medical terms
        stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by', 'this', 'that', 'these', 'those', 'are', 'was', 'were', 'be', 'been', 'being', 'can', 'could', 'should', 'would', 'will', 'what', 'when', 'where', 'who', 'how', 'does', 'do', 'did', 'has', 'have', 'had'}
        
        # Extract ALL meaningful terms including short ones for medical context
        query_words = query.split()
        query_terms = []
        
        for word in query_words:
            word_clean = word.lower().strip('?.,!;:')
            # Include medical/drug names (often capitalized or unique)
            if len(word_clean) > 2 and word_clean not in stop_words:
                query_terms.append(word_clean)
            # Also include the original case for drug names
            if word[0].isupper() and len(word) > 3:
                query_terms.append(word.strip('?.,!;:'))
        
        # Remove duplicates while preserving order
        query_terms = list(dict.fromkeys(query_terms))
        
        for similarity, idx, sentence in sentence_similarities[:num_snippets * 2]:
            # Avoid duplicate positions
            if idx in used_positions:
                continue
            
            # Mark surrounding positions as used
            for j in range(max(0, idx-1), min(len(sentences), idx+2)):
                used_positions.add(j)
            
            # Enhanced highlighting with stem matching
            highlighted = sentence
            
            # Track what we've highlighted to avoid duplicates
            highlighted_positions = []
            
            # Sort terms by length (longest first) to avoid partial matches
            sorted_terms = sorted(query_terms, key=len, reverse=True)
            
            for term in sorted_terms:
                term_lower = term.lower()
                
                # Find all occurrences with variations
                # Pattern 1: Exact match (case insensitive)
                pattern1 = re.compile(r'\b(' + re.escape(term) + r')\b', re.IGNORECASE)
                
                # Pattern 2: Word starts with term (catches plurals, conjugations)
                if len(term_lower) > 4:
                    pattern2 = re.compile(r'\b(' + re.escape(term_lower) + r'\w*)\b', re.IGNORECASE)
                else:
                    pattern2 = None
                
                # Pattern 3: Stem matching (first 4-5 chars for medical terms)
                stem_patterns = []
                if len(term_lower) >= 5:
                    stem = term_lower[:5]
                    stem_patterns.append(re.compile(r'\b(' + re.escape(stem) + r'\w{1,15})\b', re.IGNORECASE))
                
                # Apply patterns in order of specificity
                for pattern in [pattern1, pattern2] + stem_patterns:
                    if pattern:
                        matches = list(pattern.finditer(highlighted))
                        for match in reversed(matches):  # Reverse to maintain positions
                            start, end = match.span(1)
                            
                            # Check if already highlighted
                            overlap = False
                            for h_start, h_end in highlighted_positions:
                                if (start >= h_start and start < h_end) or (end > h_start and end <= h_end):
                                    overlap = True
                                    break
                            
                            if not overlap:
                                matched_text = highlighted[start:end]
                                # Only highlight if it's semantically relevant
                                if len(matched_text) >= 3:
                                    highlighted = (
                                        highlighted[:start] + 
                                        f'<mark style="background-color: #ffeb3b; padding: 2px 4px; border-radius: 3px; font-weight: 500;">{matched_text}</mark>' +
                                        highlighted[end:]
                                    )
                                    # Update positions (account for added HTML)
                                    html_added = len('<mark style="background-color: #ffeb3b; padding: 2px 4px; border-radius: 3px; font-weight: 500;">') + len('</mark>')
                                    highlighted_positions.append((start, end + html_added))
            
            # IMPROVED: Smart truncation that properly handles HTML tags
            if len(highlighted) > snippet_length:
                # First try to find a sentence boundary
                last_period = highlighted[:snippet_length].rfind('.')
                
                if last_period > snippet_length * 0.6:
                    # Good sentence boundary found
                    truncate_point = last_period + 1
                else:
                    # No good sentence boundary, need to be careful with HTML
                    truncate_point = snippet_length
                    
                    # Check if we're inside or near a <mark> tag
                    # Find all mark tag positions
                    mark_starts = [m.start() for m in re.finditer(r'<mark[^>]*>', highlighted)]
                    mark_ends = [m.end() for m in re.finditer(r'</mark>', highlighted)]
                    
                    # Check if truncate point is inside a tag
                    for mark_start in mark_starts:
                        if mark_start < truncate_point < mark_start + 100:  # Inside opening tag
                            # Move to before this tag starts
                            truncate_point = mark_start
                            break
                    
                    # If we're between opening and closing mark, extend to after closing
                    for i, mark_start in enumerate(mark_starts):
                        if i < len(mark_ends):
                            mark_end = mark_ends[i]
                            if mark_start < truncate_point < mark_end:
                                # We're inside marked text, extend to include closing tag
                                truncate_point = mark_end
                                break
                
                # Truncate at safe point
                highlighted = highlighted[:truncate_point]
                
                # Ensure all opened <mark> tags are closed
                open_marks = highlighted.count('<mark')
                close_marks = highlighted.count('</mark>')
                
                if open_marks > close_marks:
                    highlighted += '</mark>' * (open_marks - close_marks)
                
                # Remove any incomplete tags at the end
                # Check if we have an incomplete opening tag
                last_opening = highlighted.rfind('<mark')
                last_closing = highlighted.rfind('>')
                
                if last_opening > last_closing:
                    # We have an incomplete opening tag, remove it
                    highlighted = highlighted[:last_opening]
                
                # Check for incomplete closing tag
                if highlighted.endswith('</mark') or highlighted.endswith('</mar') or \
                   highlighted.endswith('</ma') or highlighted.endswith('</m') or highlighted.endswith('</'):
                    # Find the last complete tag
                    last_complete = max(
                        highlighted.rfind('</mark>'),
                        highlighted.rfind('>') if not highlighted[highlighted.rfind('>'):].startswith('></') else -1
                    )
                    if last_complete > 0:
                        highlighted = highlighted[:last_complete + 1]
                
                # Add ellipsis if we actually truncated content
                if truncate_point < len(sentence):
                    highlighted += "..."
            
            snippets.append(highlighted)
            
            if len(snippets) >= num_snippets:
                break
        
        return snippets
    
    def save_cache(self, directory: str):
        """Save embeddings to cache"""
        pass  # Caching disabled to avoid pickle issues
    
    def load_cache(self, directory: str) -> bool:
        """Try to load cached embeddings"""
        return False  # Caching disabled to avoid pickle issues
    
    def index_documents(self, directory: str, file_pattern: str = "") -> int:
        """Index all Word Q&A documents in a directory"""
        self.documents = []
        directory_path = Path(directory)
        
        # Find all .docx files
        files = list(directory_path.rglob('*.docx'))
        
        # Filter out temporary Word files (starting with ~$)
        files = [f for f in files if not f.name.startswith('~$')]
        
        # Apply file pattern filter if specified
        if file_pattern:
            files = [f for f in files if file_pattern.lower() in f.name.lower()]
        
        if not files:
            st.warning("No Word documents found in the directory")
            return 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process each file
        for idx, file_path in enumerate(files):
            status_text.text(f"Processing: {file_path.name} ({idx+1}/{len(files)})")
            
            # Extract text
            text = self.extract_text_from_docx(str(file_path))
            
            if not text.strip():
                st.warning(f"No text extracted from {file_path.name}")
                continue
            
            # Get file metadata
            file_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
            word_count = len(text.split())
            
            # Try to extract Q&A pairs (optional)
            qa_pairs = self.extract_qa_pairs(text)
            
            # Create document object
            doc = QADocument(
                filename=file_path.name,
                full_text=text,
                file_path=str(file_path),
                file_modified_date=file_modified.strftime('%Y-%m-%d'),
                word_count=word_count,
                qa_pairs=qa_pairs if qa_pairs else None
            )
            
            self.documents.append(doc)
            progress_bar.progress((idx + 1) / len(files))
        
        if not self.documents:
            st.error("No documents were successfully processed")
            progress_bar.empty()
            status_text.empty()
            return 0
        
        status_text.text("Creating document embeddings...")
        
        # Create embeddings for entire documents
        document_texts = [doc.full_text for doc in self.documents]
        embeddings = self.model.encode(
            document_texts, 
            show_progress_bar=True, 
            batch_size=8,
            convert_to_numpy=True
        )
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        progress_bar.empty()
        status_text.empty()
        
        return len(self.documents)
    
    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> List[Tuple[QADocument, float]]:
        """Search for most relevant Q&A documents"""
        if self.index is None or len(self.documents) == 0:
            return []
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        k = min(top_k, len(self.documents))
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Return results above threshold
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if score >= min_score:
                results.append((self.documents[idx], float(score)))
        
        return results

# Medical model options
MEDICAL_MODELS = {
    'PubMedBERT (Search-Optimized) - RECOMMENDED': 'pritamdeka/S-PubMedBert-MS-MARCO',
    'PubMedBERT (Base)': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
    'BioBERT v1.2': 'dmis-lab/biobert-base-cased-v1.2',
    'SciBERT': 'allenai/scibert_scivocab_uncased',
    'General (Fast & Lightweight)': 'all-MiniLM-L6-v2'
}

# Example questions organized by category
EXAMPLE_QUESTIONS = {
    "Indication & Mechanism": [
        "What is the approved indication for [drug name], and in which patient populations has it been studied?",
        "How does the mechanism of action of [drug name] differ from traditional treatments?",
    ],
    "Efficacy & Clinical Data": [
        "What clinical trial data support the efficacy of [drug name]?",
        "How does [drug name] compare to standard treatment in terms of efficacy outcomes?",
        "What endpoints were used to assess functional outcomes in [drug name] clinical trials?",
    ],
    "Dosing & Administration": [
        "What dosing regimens of [drug name] have been evaluated in clinical studies?",
        "Are there specific age or weight considerations for dosing [drug name]?",
        "How and when should [drug name] be taken each day?",
        "Is dose tapering required when discontinuing [drug name] therapy?",
    ],
    "Safety & Adverse Events": [
        "What are the most common adverse events reported with [drug name] in clinical trials?",
        "What is known about the long-term safety profile of [drug name]?",
        "What side effects should I watch for while taking [drug name]?",
        "How does [drug name] impact growth, bone health, and endocrine parameters?",
    ],
    "Drug Interactions & Monitoring": [
        "Are there any known drug-drug interactions with [drug name]?",
        "What monitoring is recommended for patients receiving [drug name]?",
        "How should patients be transitioned from conventional treatments to [drug name]?",
    ],
    "Special Populations": [
        "Are there data on the use of [drug name] in patients with hepatic or renal impairment?",
        "Has [drug name] been studied in pediatric populations?",
        "Will taking [drug name] affect my child's growth or long-term health?",
    ],
    "Pharmacology": [
        "What is known about the pharmacokinetics and metabolism of [drug name]?",
        "What is the effect of [drug name] on immune function and infection risk?",
        "Does [drug name] have [specific activity], and how clinically relevant is this?",
    ],
    "Practical Information": [
        "What is [drug name], and how is it different from other medications?",
        "What benefits can I expect from taking [drug name]?",
        "Are there any contraindications or important precautions associated with [drug name] use?",
    ]
}

# Categorize questions by audience
HCP_QUESTIONS = {
    "Indication & Mechanism",
    "Efficacy & Clinical Data",
    "Drug Interactions & Monitoring",
    "Pharmacology"
}

PATIENT_QUESTIONS = {
    "Safety & Adverse Events",
    "Special Populations",
    "Dosing & Administration",
    "Practical Information"
}

# Patient-friendly terms mapping
PATIENT_FRIENDLY_TERMS = {
    'adverse events': 'side effects',
    'adverse event': 'side effect',
    'contraindications': 'reasons you should not take this',
    'contraindication': 'reason you should not take this',
    'pharmacokinetics': 'how the medicine works in your body',
    'pharmacodynamics': 'how the medicine affects your body',
    'hepatic': 'liver',
    'renal': 'kidney',
    'pediatric': 'children',
    'efficacy': 'how well it works',
    'administer': 'take',
    'administration': 'taking',
    'dosing regimen': 'how to take it',
    'drug-drug interaction': 'how it works with other medicines',
    'drug interaction': 'how it works with other medicines',
    'monitoring': 'regular check-ups',
    'metabolism': 'how your body processes it',
    'therapeutic': 'treatment',
    'titration': 'dose adjustment',
    'bioavailability': 'how much medicine gets into your blood',
    'half-life': 'how long the medicine stays in your body',
    'clinical trial': 'research study',
    'clinical endpoint': 'study result',
    'prophylaxis': 'prevention'
}

def detect_query_audience(query: str) -> str:
    """
    Detect if a query is from an HCP or patient based on language complexity
    """
    query_lower = query.lower()
    
    # Strong HCP indicators (technical medical terms)
    strong_hcp_terms = [
        'pharmacokinetics', 'pharmacodynamics', 'mechanism of action', 
        'bioavailability', 'half-life', 'therapeutic index', 'metabolism',
        'titration', 'clinical endpoint', 'efficacy outcomes', 'pharmacology',
        'drug-drug interaction', 'monitoring parameters', 'dosing regimen',
        'adverse events', 'contraindication', 'hepatic impairment', 'renal impairment'
    ]
    
    # Moderate HCP indicators
    moderate_hcp_terms = [
        'clinical trial', 'study', 'trial data', 'evidence', 'efficacy',
        'clinical data', 'endpoints', 'parameters', 'regimen'
    ]
    
    # Strong patient indicators (personal/direct language)
    strong_patient_terms = [
        'my child', 'my son', 'my daughter', 'i take', 'should i', 'can i',
        'will it affect me', 'is it safe for me', 'what if i miss', 'how do i take',
        'my doctor', 'i am taking', 'i have been', 'for my', 'will i'
    ]
    
    # Moderate patient indicators (simple questions)
    moderate_patient_terms = [
        'plain language', 'simple terms', 'easy to understand',
        'side effects', 'safe to take', 'when should i'
    ]
    
    # Check for explicit audience request
    if any(term in query_lower for term in ['patient-friendly', 'caregiver', 'parent', 'in plain language']):
        return 'patient'
    if any(term in query_lower for term in ['hcp', 'healthcare provider', 'physician', 'for prescribers']):
        return 'hcp'
    
    # Count indicators with weights
    strong_hcp_count = sum(1 for term in strong_hcp_terms if term in query_lower)
    moderate_hcp_count = sum(1 for term in moderate_hcp_terms if term in query_lower)
    strong_patient_count = sum(1 for term in strong_patient_terms if term in query_lower)
    moderate_patient_count = sum(1 for term in moderate_patient_terms if term in query_lower)
    
    # Weight the scores
    hcp_score = (strong_hcp_count * 3) + moderate_hcp_count
    patient_score = (strong_patient_count * 3) + moderate_patient_count
    
    # Strong HCP terms always win
    if strong_hcp_count > 0:
        return 'hcp'
    
    # Strong patient terms always win (if no strong HCP terms)
    if strong_patient_count > 0:
        return 'patient'
    
    # Compare weighted scores
    if hcp_score > patient_score:
        return 'hcp'
    elif patient_score > hcp_score:
        return 'patient'
    
    # If tied, check question complexity as tiebreaker
    # Remove common words
    words = query_lower.split()
    content_words = [w for w in words if len(w) > 3 and w not in ['what', 'how', 'when', 'where', 'which', 'does', 'this', 'that', 'with', 'from', 'have', 'been']]
    
    # Check average word length of content words
    if content_words:
        avg_word_length = sum(len(word) for word in content_words) / len(content_words)
        # Longer average word length suggests HCP (medical terminology tends to be longer)
        return 'hcp' if avg_word_length >= 7 else 'patient'
    
    # Default to patient mode for safety
    return 'patient'

def simplify_for_patients(text: str) -> str:
    """
    Replace medical jargon with patient-friendly language
    """
    simplified = text
    # Sort by length (longest first) to avoid partial replacements
    sorted_terms = sorted(PATIENT_FRIENDLY_TERMS.items(), key=lambda x: len(x[0]), reverse=True)
    
    for medical_term, friendly_term in sorted_terms:
        # Case-insensitive replacement with word boundaries
        pattern = re.compile(r'\b' + re.escape(medical_term) + r'\b', re.IGNORECASE)
        simplified = pattern.sub(friendly_term, simplified)
    
    return simplified

def main():
    st.set_page_config(
        page_title="Pharma Q&A Document Finder", 
        page_icon="💊", 
        layout="wide"
    )
    
    st.title("💊 Pharma Q&A Document Finder")
    st.markdown("Find the most relevant Q&A documents for HCP and public queries using semantic search")
    
    # Initialize session state
    if 'searcher' not in st.session_state:
        st.session_state.searcher = None
        st.session_state.indexed = False
        st.session_state.current_model = None
        st.session_state.last_results = []
        st.session_state.last_query = ""
        st.session_state.temp_dir = None
        st.session_state.forced_audience = None
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        st.subheader("🤖 Model Selection")
        selected_model_name = st.selectbox(
            "Choose a model:",
            options=list(MEDICAL_MODELS.keys()),
            index=0,
            help="PubMedBERT (Search-Optimized) is best for pharma documents"
        )
        
        selected_model = MEDICAL_MODELS[selected_model_name]
        
        with st.expander("ℹ️ Model Info"):
            st.markdown("""
            **PubMedBERT (Search-Optimized)**: Best choice - trained on medical 
            literature and fine-tuned specifically for search/retrieval tasks.
            
            **General (Fast)**: Faster but less accurate for medical terminology.
            
            **Semantic Matching**: Uses sentence-level similarity for ranking.
            Enhanced highlighting catches variations like "study/studied/studies".
            """)
        
        # Initialize model
        if st.session_state.searcher is None or st.session_state.current_model != selected_model:
            if st.button("🔄 Load Model", type="primary", use_container_width=True):
                with st.spinner("Loading model... (first time will download ~400MB)"):
                    st.session_state.searcher = PharmaQASearcher(model_name=selected_model)
                    st.session_state.current_model = selected_model
                    st.session_state.indexed = False
                    st.success(f"✅ Model loaded!")
        
        if st.session_state.searcher is not None:
            st.success(f"✅ {selected_model_name.split(' - ')[0]}")
        
        st.divider()
        
        # Document source selection
        st.subheader("📁 Document Source")
        
        source_option = st.radio(
            "Choose document source:",
            ["Upload Files", "Local Directory"],
            help="Upload files directly or point to a local folder"
        )
        
        if source_option == "Upload Files":
            # File upload section
            st.markdown("**📤 Upload Q&A Documents**")
            
            # SharePoint tip
            with st.expander("💡 Uploading from SharePoint/OneDrive?"):
                st.markdown("""
                **If you're having trouble uploading SharePoint files:**
                
                1. **Download to your computer first:**
                   - Open the file in SharePoint
                   - Click "..." menu → Download
                   - Then upload here
                
                2. **Or use batch download:**
                   - Select all files in SharePoint
                   - Click Download
                   - Extract the ZIP
                   - Upload files from the extracted folder
                
                3. **Avoid:** Don't upload directly from SharePoint's "Open in Desktop App" - 
                   this can cause permission issues.
                """)
            
            uploaded_files = st.file_uploader(
                "Upload Word documents:",
                type=['docx'],
                accept_multiple_files=True,
                help="Upload .docx files to search (max 200MB each)"
            )
            
            if uploaded_files:
                st.info(f"📄 {len(uploaded_files)} files selected")
                
                if st.session_state.searcher is not None:
                    if st.button("📚 Index Uploaded Documents", type="primary", use_container_width=True):
                        # Clean up previous temp directory
                        if st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
                            shutil.rmtree(st.session_state.temp_dir)
                        
                        # Create new temp directory
                        st.session_state.temp_dir = tempfile.mkdtemp()
                        
                        # Save uploaded files with better error handling
                        saved_count = 0
                        failed_files = []
                        
                        for uploaded_file in uploaded_files:
                            try:
                                # Clean filename - remove special characters from SharePoint
                                clean_filename = uploaded_file.name
                                # Remove common SharePoint metadata patterns
                                clean_filename = re.sub(r'[<>:"|?*]', '', clean_filename)
                                # Remove any path separators
                                clean_filename = os.path.basename(clean_filename)
                                # Limit filename length
                                if len(clean_filename) > 200:
                                    name, ext = os.path.splitext(clean_filename)
                                    clean_filename = name[:196] + ext
                                
                                file_path = os.path.join(st.session_state.temp_dir, clean_filename)
                                
                                # Write file in binary mode
                                with open(file_path, "wb") as f:
                                    # Get bytes and write
                                    file_bytes = uploaded_file.getbuffer()
                                    f.write(file_bytes)
                                
                                saved_count += 1
                            except Exception as e:
                                failed_files.append((uploaded_file.name, str(e)))
                        
                        # Show summary
                        if failed_files:
                            st.warning(f"⚠️ {len(failed_files)} file(s) failed to upload")
                            with st.expander("View failed files"):
                                for fname, error in failed_files:
                                    st.error(f"**{fname}**: {error}")
                        
                        if saved_count > 0:
                            st.success(f"✅ Saved {saved_count} files")
                            
                            # Index the documents
                            with st.spinner("Indexing uploaded documents..."):
                                num_docs = st.session_state.searcher.index_documents(st.session_state.temp_dir)
                                if num_docs > 0:
                                    st.session_state.indexed = True
                                    st.success(f"✅ Indexed {num_docs} documents!")
                                else:
                                    st.error("❌ No documents were processed")
                        else:
                            st.error("❌ No files could be saved")
                else:
                    st.warning("⚠️ Please load a model first")
        
        else:
            # Local directory section
            st.markdown("**📂 Local Directory**")
            
            with st.expander("💡 How to find your folder path"):
                st.markdown("""
                **On Mac:**
                1. Open Finder and go to your documents folder
                2. Right-click the folder
                3. Hold `Option` key and select "Copy as Pathname"
                4. Paste below
                
                **On Windows:**
                1. Open File Explorer and go to your folder
                2. Click the address bar at the top
                3. Copy the path (e.g., `C:\\Users\\Name\\Documents\\PharmaQA`)
                4. Paste below
                """)
            
            directory = st.text_input(
                "Directory Path:",
                value=os.path.expanduser("~/Documents/PharmaQA"),
                help="Paste the full path to your Q&A documents folder",
                placeholder="/Users/yourname/Documents/PharmaQA"
            )
            
            # Validate and show folder contents preview
            if directory and os.path.exists(directory):
                docx_files = list(Path(directory).rglob('*.docx'))
                docx_files = [f for f in docx_files if not f.name.startswith('~$')]
                if docx_files:
                    st.success(f"✅ Found {len(docx_files)} .docx files")
                else:
                    st.warning("⚠️ No .docx files found in this folder")
            elif directory:
                st.error("❌ Folder does not exist")
            
            file_pattern = st.text_input(
                "File Filter (optional):",
                value="",
                placeholder="e.g., 'product_name' or 'Q&A'",
                help="Only index files containing this text in filename"
            )
            
            # Index button for local directory
            if st.session_state.searcher is not None:
                if st.button("📚 Index Documents", type="primary", use_container_width=True):
                    if not os.path.exists(directory):
                        st.error("❌ Directory does not exist!")
                    else:
                        with st.spinner("Indexing Q&A documents..."):
                            num_docs = st.session_state.searcher.index_documents(
                                directory,
                                file_pattern=file_pattern
                            )
                            if num_docs > 0:
                                st.session_state.indexed = True
                                st.success(f"✅ Indexed {num_docs} documents!")
                            else:
                                st.error("❌ No documents found or processed")
        
        # Statistics
        if st.session_state.indexed:
            st.divider()
            st.subheader("📊 Index Stats")
            
            total_docs = len(st.session_state.searcher.documents)
            total_words = sum(doc.word_count for doc in st.session_state.searcher.documents)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", total_docs)
            with col2:
                st.metric("Total Words", f"{total_words:,}")
            
            # Document list
            with st.expander("📄 Indexed Documents"):
                for doc in st.session_state.searcher.documents:
                    st.markdown(f"**{doc.filename}**")
                    st.caption(f"Modified: {doc.file_modified_date} | Words: {doc.word_count:,}")
                    if doc.qa_pairs:
                        st.caption(f"Q&A Pairs detected: {len(doc.qa_pairs)}")
                    st.divider()
    
    # Main search interface
    if st.session_state.indexed:
        
        # Search section
        st.markdown("### 🔎 Search Query")
        
        # Initialize example query holder
        if 'example_to_load' not in st.session_state:
            st.session_state.example_to_load = ""
        
        # Example questions section
        with st.expander("💡 Example Questions - Click to Use", expanded=False):
            st.markdown("**Note:** Replace `[drug name]` with your actual product name")
            
            # Create tabs for different categories
            tab_names = list(EXAMPLE_QUESTIONS.keys())
            tabs = st.tabs(tab_names)
            
            for tab, category in zip(tabs, tab_names):
                with tab:
                    # Show audience badge
                    if category in HCP_QUESTIONS:
                        st.caption("👨‍⚕️ Healthcare Professional Questions")
                    elif category in PATIENT_QUESTIONS:
                        st.caption("🏥 Patient/Caregiver Questions")
                    
                    for idx, question in enumerate(EXAMPLE_QUESTIONS[category]):
                        button_key = f"ex_{category}_{idx}"
                        if st.button(question, key=button_key, use_container_width=True):
                            st.session_state.example_to_load = question
                            st.rerun()
        
        # Use form to encapsulate the query input
        with st.form(key="search_form", clear_on_submit=False):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Use the example if one was selected
                initial_value = st.session_state.example_to_load
                query = st.text_area(
                    "Enter your query:",
                    value=initial_value,
                    placeholder="Example: What is the approved indication for [drug name]?\nExample: What are the side effects?",
                    height=100,
                    label_visibility="collapsed"
                )
                # Clear the example after loading
                if initial_value:
                    st.session_state.example_to_load = ""
            
            with col2:
                st.markdown("**Search Options**")
                
                # Audience selector
                audience_mode = st.radio(
                    "Audience:",
                    ["Auto-detect", "Patient/Caregiver", "Healthcare Professional"],
                    help="Choose how results should be presented"
                )
                
                num_results = st.number_input("Results:", 1, 10, 3)
                min_score = st.slider("Min Score:", 0.0, 1.0, 0.2, 0.05)
            
            # Submit button for the form
            submitted = st.form_submit_button("🔍 Find Documents", type="primary", use_container_width=True)
            
            if submitted:
                if query.strip():
                    # Store audience preference
                    if audience_mode == "Patient/Caregiver":
                        st.session_state.forced_audience = 'patient'
                    elif audience_mode == "Healthcare Professional":
                        st.session_state.forced_audience = 'hcp'
                    else:
                        st.session_state.forced_audience = None
                        
                    with st.spinner("Searching..."):
                        results = st.session_state.searcher.search(
                            query, 
                            top_k=num_results,
                            min_score=min_score
                        )
                        st.session_state.last_results = results
                        st.session_state.last_query = query
                else:
                    st.warning("Please enter a query")
        
        # Display results
        if st.session_state.last_results:
            results = st.session_state.last_results
            
            # Detect audience
            if st.session_state.forced_audience:
                audience = st.session_state.forced_audience
            else:
                audience = detect_query_audience(st.session_state.last_query)
            
            st.markdown("---")
            
            # Show audience indicator
            if audience == 'patient':
                st.info("🏥 **Patient/Caregiver Mode**: Results shown in plain language")
            else:
                st.info("👨‍⚕️ **Healthcare Professional Mode**: Results shown with medical terminology")
            
            st.markdown(f"### 📋 Top {len(results)} Most Relevant Documents")
            
            if len(results) == 0:
                st.info("💡 No documents met the minimum relevance threshold. Try:\n- Lowering the minimum score\n- Rephrasing your query\n- Using more specific medical terms")
            
            for idx, (doc, score) in enumerate(results, 1):
                # Relevance indicator
                if score >= 0.7:
                    indicator = "🟢 High"
                    color = "green"
                elif score >= 0.5:
                    indicator = "🟡 Medium"
                    color = "orange"
                else:
                    indicator = "🟠 Low"
                    color = "red"
                
                # Document card
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.markdown(f"### {idx}. {doc.filename}")
                    with col2:
                        st.markdown(f"<div style='text-align: right; padding-top: 10px;'>{indicator}<br><span style='color: {color}; font-size: 20px; font-weight: bold;'>{score:.1%}</span></div>", unsafe_allow_html=True)
                    
                    # Metadata
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.caption(f"📅 Modified: {doc.file_modified_date}")
                    with col2:
                        st.caption(f"📝 {doc.word_count:,} words")
                    with col3:
                        if doc.qa_pairs:
                            st.caption(f"❓ {len(doc.qa_pairs)} Q&A pairs")
                    
                    # Get relevant snippets
                    snippets = st.session_state.searcher.get_relevant_snippets(
                        doc.full_text, 
                        st.session_state.last_query,
                        num_snippets=3
                    )
                    
                    # Apply patient-friendly language if needed
                    if audience == 'patient':
                        snippets = [simplify_for_patients(snippet) for snippet in snippets]
                    
                    # Show highlighted snippets
                    if snippets:
                        if audience == 'patient':
                            st.markdown("**📌 Key Information (in plain language):**")
                        else:
                            st.markdown("**📌 Most Relevant Excerpts:**")
                            
                        for snippet_idx, snippet in enumerate(snippets, 1):
                            st.markdown(f"{snippet_idx}. {snippet}", unsafe_allow_html=True)
                            if snippet_idx < len(snippets):
                                st.markdown("")  # Add spacing
                    
                    # Content preview
                    with st.expander("👁️ View Full Document", expanded=False):
                        # Show highlighted full content
                        highlighted_text = st.session_state.searcher.highlight_text(
                            doc.full_text, 
                            st.session_state.last_query,
                            max_length=2000
                        )
                        
                        # Apply patient-friendly language if needed
                        if audience == 'patient':
                            highlighted_text = simplify_for_patients(highlighted_text)
                        
                        st.markdown(highlighted_text, unsafe_allow_html=True)
                        
                        # Show Q&A pairs if detected
                        if doc.qa_pairs:
                            st.markdown("---")
                            st.markdown("**Detected Q&A Pairs:**")
                            for q, a in doc.qa_pairs[:5]:  # Show first 5
                                with st.container():
                                    q_display = simplify_for_patients(q) if audience == 'patient' else q
                                    a_display = simplify_for_patients(a) if audience == 'patient' else a
                                    st.markdown(f"**Q:** {q_display}")
                                    st.markdown(f"**A:** {a_display[:300]}{'...' if len(a_display) > 300 else ''}")
                                    st.divider()
                            if len(doc.qa_pairs) > 5:
                                st.caption(f"+ {len(doc.qa_pairs) - 5} more Q&A pairs")
                    
                    # Action buttons
                    col1, col2, col3 = st.columns([2, 2, 3])
                    with col1:
                        # Download button for uploaded files
                        if doc.file_path.startswith(tempfile.gettempdir()):
                            # This is an uploaded file
                            try:
                                with open(doc.file_path, 'rb') as f:
                                    st.download_button(
                                        f"⬇️ Download",
                                        f,
                                        file_name=doc.filename,
                                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                        key=f"download_{idx}"
                                    )
                            except:
                                st.caption("Download unavailable")
                        else:
                            # This is a local file
                            if st.button(f"📂 Open Document", key=f"open_{idx}"):
                                try:
                                    import subprocess
                                    import platform
                                    
                                    if platform.system() == 'Darwin':  # macOS
                                        subprocess.call(['open', doc.file_path])
                                    elif platform.system() == 'Windows':
                                        os.startfile(doc.file_path)
                                    else:  # linux
                                        subprocess.call(['xdg-open', doc.file_path])
                                    st.success(f"Opening {doc.filename}")
                                except Exception as e:
                                    st.error(f"Could not open file: {e}")
                    
                    with col2:
                        if st.button(f"📋 Copy Path", key=f"copy_{idx}"):
                            st.code(doc.file_path, language=None)
                    
                    st.markdown("---")
            
            # Export results
            with st.expander("📊 Export Search Results"):
                export_text = f"PHARMA Q&A SEARCH RESULTS\n"
                export_text += f"Query: {st.session_state.last_query}\n"
                export_text += f"Audience: {'Patient/Caregiver' if audience == 'patient' else 'Healthcare Professional'}\n"
                export_text += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                export_text += f"Results: {len(results)}\n"
                export_text += "=" * 80 + "\n\n"
                
                for idx, (doc, score) in enumerate(results, 1):
                    export_text += f"{idx}. {doc.filename}\n"
                    export_text += f"   Relevance: {score:.2%}\n"
                    export_text += f"   Modified: {doc.file_modified_date}\n"
                    export_text += f"   Path: {doc.file_path}\n"
                    
                    # Add snippets to export
                    snippets = st.session_state.searcher.get_relevant_snippets(
                        doc.full_text, 
                        st.session_state.last_query,
                        num_snippets=3
                    )
                    
                    # Apply patient-friendly language
                    if audience == 'patient':
                        snippets = [simplify_for_patients(snippet) for snippet in snippets]
                    
                    if snippets:
                        export_text += f"   Excerpts:\n"
                        for snippet in snippets:
                            # Remove HTML tags for plain text export
                            clean_snippet = re.sub(r'<[^>]+>', '', snippet)
                            export_text += f"     - {clean_snippet}\n"
                    
                    export_text += "-" * 80 + "\n\n"
                
                st.download_button(
                    "⬇️ Download Results",
                    export_text,
                    file_name=f"qa_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        elif st.session_state.last_query:
            st.info("👆 Enter a query and click 'Find Documents' to search")
    
    else:
        # Welcome screen
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 How It Works
            
            1. **Load Model** - Choose and load a medical AI model
            2. **Choose Source** - Upload files or point to local directory
            3. **Index Documents** - Process all Word documents
            4. **Search** - Enter any query or use example questions
            5. **Review** - See ranked documents with highlighted matches
            
            ### ✨ Features
            
            - 🧠 **Semantic Search** - Understands meaning, not just keywords
            - 💊 **Medical Domain** - Trained on pharmaceutical literature
            - 🔍 **Enhanced Highlighting** - Catches variations and stems
            - 💡 **Example Questions** - 30+ pre-written HCP and patient questions
            - 🏥 **Patient Mode** - Auto-simplifies medical jargon for patients
            - 👨‍⚕️ **HCP Mode** - Preserves medical terminology for professionals
            - ⚡ **Fast** - Optimized for speed
            - 🔒 **100% Local** - No data leaves your machine
            - 📤 **File Upload** - Works with uploaded or local files
            - 📊 **Relevance Scoring** - See how well documents match
            - 🪟 **Cross-Platform** - Works on Mac and Windows
            """)
        
        with col2:
            st.markdown("""
            ### 💡 Example Question Categories
            
            **👨‍⚕️ For Healthcare Professionals:**
            - Indication & Mechanism (2 questions)
            - Efficacy & Clinical Data (3 questions)
            - Drug Interactions & Monitoring (3 questions)
            - Pharmacology (3 questions)
            
            **🏥 For Patients/Caregivers:**
            - Safety & Adverse Events (4 questions)
            - Special Populations (3 questions)
            - Dosing & Administration (4 questions)
            - Practical Information (3 questions)
            
            ### 🔄 Language Simplification
            
            **Patient Mode automatically replaces:**
            - "adverse events" → "side effects"
            - "contraindications" → "reasons you should not take this"
            - "hepatic" → "liver"
            - "renal" → "kidney"
            - "efficacy" → "how well it works"
            - And 15+ more medical terms!
            """)
        
        st.markdown("---")
        
        if st.session_state.searcher is None:
            st.info("👈 **Step 1:** Load a model from the sidebar to begin")
        else:
            st.info("👈 **Step 2:** Upload documents or select local directory in the sidebar")

if __name__ == "__main__":
    main()
