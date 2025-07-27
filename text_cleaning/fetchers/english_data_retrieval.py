import boto3
import re
from collections import defaultdict
import pandas as pd
import os
from itertools import chain
from datetime import datetime
from datasets import load_dataset
from tqdm import tqdm

# ==== User Configuration ====
AWS_ACCESS_KEY= ""
AWS_SECRET_KEY = ""
S3_URI = ""

# Processing parameters
CHUNKS_PER_FILE = 200000  # Documents per file (increased for bigger chunks)
MIN_TOKEN_COUNT = 50  # Minimum tokens per document
TARGET_TOKENS = 110_000_000_000  # 110 billion tokens
MAX_TOKENS_PER_TOPIC = 1_500_000_000  # 1.5B tokens max per topic (reduced for MORE diversity)

# Date filter - April 21, 2025 and later
MIN_DATE = datetime(2025, 4, 21)

# FineWeb snapshots - using all available 2025 snapshots for post-April data
SNAPSHOTS = [
    "CC-MAIN-2025-26",  # Latest available snapshot (likely June 2025)
    "CC-MAIN-2025-21",  # May 2025
    "CC-MAIN-2025-18",  # May 2025
    "CC-MAIN-2025-13",  # April 2025
    "CC-MAIN-2025-08",  # March 2025 (might have some April data)
    "CC-MAIN-2025-05",  # February 2025 (might have some April data)
]


def classify_topic(text, url=""):
    """Classify document topic based on content and URL"""
    if not text:
        return "unknown"

    # Convert to lowercase for matching
    text_lower = text.lower()
    url_lower = url.lower() if url else ""

    # Define topic keywords and URL patterns
    topic_patterns = {
        "politics": [
            "election", "government", "president", "congress", "senate", "political", "vote", "democracy",
            "republican", "democrat", "policy", "legislation", "campaign", "politician", "parliament",
            "minister", "governor", "mayor", "/politics/", "political"
        ],
        "technology": [
            "software", "computer", "internet", "digital", "tech", "programming", "code", "artificial intelligence",
            "machine learning", "blockchain", "cryptocurrency", "startup", "silicon valley", "/tech/", "github", "api"
        ],
        "science": [
            "research", "study", "scientific", "experiment", "discovery", "science", "biology", "chemistry",
            "physics", "medicine", "medical", "health", "doctor", "disease", "treatment", "/science/", "journal"
        ],
        "history": [
            "historical", "history", "ancient", "century", "war", "battle", "empire", "civilization",
            "museum", "archaeological", "heritage", "past", "timeline", "/history/", "historic"
        ],
        "business": [
            "business", "company", "corporation", "market", "economy", "financial", "investment", "stock",
            "revenue", "profit", "ceo", "industry", "commerce", "/business/", "entrepreneur", "startup"
        ],
        "sports": [
            "sport", "game", "team", "player", "match", "championship", "league", "football", "basketball",
            "baseball", "soccer", "tennis", "golf", "/sports/", "athlete", "tournament"
        ],
        "entertainment": [
            "movie", "film", "music", "celebrity", "actor", "actress", "entertainment", "hollywood",
            "concert", "album", "song", "tv show", "series", "/entertainment/", "cinema", "theater"
        ],
        "education": [
            "education", "school", "university", "college", "student", "teacher", "learning", "academic",
            "curriculum", "degree", "scholarship", "/education/", "classroom", "lecture"
        ],
        "health": [
            "health", "medical", "medicine", "doctor", "hospital", "treatment", "disease", "patient",
            "healthcare", "wellness", "fitness", "/health/", "clinic", "therapy"
        ],
        "finance": [
            "finance", "money", "bank", "investment", "loan", "credit", "debt", "insurance",
            "financial", "economy", "budget", "/finance/", "trading", "mortgage"
        ],
        "travel": [
            "travel", "tourism", "vacation", "hotel", "flight", "destination", "trip", "adventure",
            "culture", "explore", "/travel/", "journey", "tourist"
        ],
        "food": [
            "food", "recipe", "cooking", "restaurant", "chef", "cuisine", "meal", "dish",
            "nutrition", "ingredient", "/food/", "culinary", "dining"
        ],
        "lifestyle": [
            "lifestyle", "fashion", "beauty", "home", "family", "relationship", "parenting", "wedding",
            "personal", "advice", "/lifestyle/", "living", "style"
        ],
        "news": [
            "breaking", "news", "report", "journalist", "media", "press", "announcement", "update",
            "headline", "story", "/news/", "correspondent", "coverage"
        ],
        "legal": [
            "legal", "law", "court", "judge", "lawyer", "attorney", "case", "trial",
            "justice", "lawsuit", "/legal/", "regulation", "legislation"
        ]
    }

    # Count matches for each topic
    topic_scores = {}

    for topic, keywords in topic_patterns.items():
        score = 0
        for keyword in keywords:
            # Check in text (give more weight to early text)
            text_sample = text_lower[:2000]  # First 2000 chars
            score += text_sample.count(keyword) * 2

            # Check in URL (strong signal)
            if keyword in url_lower:
                score += 5

        if score > 0:
            topic_scores[topic] = score

    # Return topic with highest score, or classify by domain patterns
    if topic_scores:
        return max(topic_scores, key=topic_scores.get)

    # Fallback: classify by domain patterns if no content matches
    domain_topics = {
        "wikipedia": "reference",
        "reddit": "discussion",
        "stackoverflow": "technology",
        "github": "technology",
        "youtube": "entertainment",
        "amazon": "business",
        "ebay": "business",
        "linkedin": "business",
        "facebook": "social",
        "twitter": "social",
        "instagram": "social",
        "pinterest": "lifestyle",
        "medium": "opinion",
        "quora": "discussion",
        "gov": "government",
        "edu": "education",
        "org": "organization"
    }

    for pattern, topic in domain_topics.items():
        if pattern in url_lower:
            return topic

    return "general"


def parse_date(date_str):
    """Parse various date formats from FineWeb"""
    if not date_str:
        return None

    # Remove timezone info if present
    date_str = date_str.replace("Z", "").replace("+00:00", "")

    # Try different date formats
    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except:
            continue
    return None


def estimate_tokens(text):
    """Rough token estimation (words * 1.3)"""
    if not text:
        return 0
    # Simple word-based estimation
    words = len(text.split())
    return int(words * 1.3)  # Approximate tokens


def is_english(text, lang_field=None):
    """Check if document is in English using the language field"""
    # Primary check: use the language column directly
    if lang_field:
        lang_lower = lang_field.lower().strip()
        # Accept 'en' or variations like 'en-us', 'en-gb', etc.
        if lang_lower == 'en' or lang_lower.startswith('en-'):
            return True
        # Also accept 'english'
        if lang_lower == 'english':
            return True

    # Fallback: if no language field, do basic text analysis
    if not text or len(text.strip()) < 50:
        return False

    # Simple English detection based on common words (fallback only)
    english_words = {'the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'you', 'that', 'he', 'was', 'for', 'on', 'are',
                     'as', 'with', 'his', 'they', 'i', 'at', 'be', 'this', 'have', 'from', 'or', 'one', 'had', 'by',
                     'word', 'but', 'not', 'what', 'all', 'were', 'we', 'when', 'your', 'can', 'said', 'there', 'each',
                     'which', 'she', 'do', 'how', 'their', 'if', 'will', 'up', 'other', 'about', 'out', 'many', 'then',
                     'them', 'these', 'so', 'some', 'her', 'would', 'make', 'like', 'into', 'time', 'has', 'two',
                     'more', 'very', 'after', 'words', 'its', 'through', 'just', 'form', 'much', 'before', 'right',
                     'too', 'any', 'old', 'should', 'where', 'same', 'man', 'me', 'without', 'such', 'here', 'take',
                     'why', 'things', 'help', 'put', 'years', 'different', 'away', 'again', 'off', 'went', 'tell',
                     'men', 'say', 'small', 'every', 'found', 'still', 'name', 'good', 'sentence', 'think', 'great'}

    # Get first 100 words for efficiency
    words = text.lower().split()[:100]
    if len(words) < 10:
        return False

    english_count = sum(1 for word in words if re.sub(r'[^a-z]', '', word) in english_words)
    return english_count / len(words) > 0.25  # At least 25% common English words


class FineWebPipeline:
    def __init__(self, aws_access_key, aws_secret_key, s3_uri):
        print("🔄 Initializing FineWeb Pipeline...")
        self.aws_access_key = aws_access_key
        self.aws_secret_key = aws_secret_key
        self.s3_uri = s3_uri

        # Statistics
        self.processed = 0
        self.kept = 0
        self.total_tokens = 0
        self.topic_tokens = defaultdict(int)  # Track tokens per topic (not domain)
        self.domain_tokens = defaultdict(int)  # Also track domains for reference
        self.file_count = 0
        self.doc_buffer = []

        # Create S3 client
        print("🔄 Creating S3 client...")
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key
        )

        # Parse S3 URI
        s3_parts = s3_uri.replace("s3://", "").split("/")
        self.bucket = s3_parts[0]
        self.key_prefix = "/".join(s3_parts[1:]) if len(s3_parts) > 1 else ""

        print(f"✅ Pipeline initialized. Target: {TARGET_TOKENS:,} tokens")
        print(f"📁 S3 Bucket: {self.bucket}")
        print(f"📁 S3 Prefix: {self.key_prefix}")

    def upload_chunk(self):
        """Upload current buffer to S3 as Parquet file"""
        if not self.doc_buffer:
            return

        filename = f"fineweb_chunk_{self.file_count:05d}.parquet"
        local_path = f"/tmp/{filename}"

        print(f"\n💾 Writing {len(self.doc_buffer)} docs to {filename}...")

        # Convert to DataFrame and save as Parquet
        df = pd.DataFrame(self.doc_buffer)

        # Optimize data types
        df['token_count'] = df['token_count'].astype('int32')
        df['language'] = df['language'].astype('category')
        df['topic'] = df['topic'].astype('category')  # Topic instead of domain
        df['domain'] = df['domain'].astype('category')

        # Write Parquet file with compression
        df.to_parquet(
            local_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )

        # Upload to S3
        s3_key = f"{self.key_prefix}/{filename}" if self.key_prefix else filename
        print(f"📤 Uploading to S3: s3://{self.bucket}/{s3_key}")

        try:
            self.s3.upload_file(local_path, self.bucket, s3_key)
            print(f"✅ Upload successful!")
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            raise
        finally:
            # Clean up local file
            if os.path.exists(local_path):
                os.remove(local_path)

        # Reset buffer
        self.doc_buffer = []
        self.file_count += 1

    def should_keep_document(self, doc):
        """Determine if document should be kept based on filters"""
        # Language filter - use the language column directly
        lang = doc.get("language", "")
        text = doc.get("text", "")

        # Primary filter: check language field first
        if not lang or not is_english(text, lang):
            return False, "language"

        # Date filter
        date_str = doc.get("date") or doc.get("timestamp")
        if not date_str:
            return False, "no_date"

        parsed_date = parse_date(date_str)
        if not parsed_date or parsed_date < MIN_DATE:
            return False, "date_too_old"

        # Token count filter
        token_count = doc.get("token_count")
        if not token_count:
            token_count = estimate_tokens(text)

        if token_count < MIN_TOKEN_COUNT:
            return False, "too_short"

        # Topic diversity filter
        url = doc.get("url", "")
        topic = classify_topic(text, url)

        if self.topic_tokens[topic] >= MAX_TOKENS_PER_TOPIC:
            return False, "topic_limit"

        return True, token_count

    def process_document(self, doc):
        """Process a single document"""
        self.processed += 1

        # Apply filters
        should_keep, reason = self.should_keep_document(doc)

        if not should_keep:
            return False

        # Document passed all filters
        token_count = reason  # reason contains token_count when should_keep=True
        text = doc.get("text", "")  # FIXED: Get text from doc
        url = doc.get("url", "")
        topic = classify_topic(text, url)  # FIXED: Now text is properly defined
        domain = url.split("//")[-1].split("/")[0] if url else "unknown"
        if domain.startswith("www."):
            domain = domain[4:]

        # Update statistics
        self.kept += 1
        self.total_tokens += token_count
        self.topic_tokens[topic] += token_count
        self.domain_tokens[domain.lower()] += token_count

        # Add metadata
        processed_doc = {
            "text": text,
            "url": url,
            "date": doc.get("date") or doc.get("timestamp"),
            "token_count": token_count,
            "topic": topic,  # Now using actual topic classification
            "domain": domain.lower(),
            "language": doc.get("language", "en"),
            "processed_at": datetime.now().isoformat()
        }

        self.doc_buffer.append(processed_doc)

        # Upload chunk if buffer is full
        if len(self.doc_buffer) >= CHUNKS_PER_FILE:
            self.upload_chunk()

        return True

    def run(self):
        """Main processing loop"""
        print("📥 Loading streaming datasets...")

        # Load all snapshots
        streams = []
        for snapshot in SNAPSHOTS:
            try:
                print(f"  - Loading snapshot: {snapshot}")
                stream = load_dataset(
                    "HuggingFaceFW/fineweb",
                    name=snapshot,
                    split="train",
                    streaming=True
                )
                streams.append(stream)
            except Exception as e:
                print(f"  ⚠️  Failed to load {snapshot}: {e}")
                continue

        if not streams:
            print("❌ No snapshots loaded successfully!")
            return

        # Combine streams
        dataset = chain(*streams)
        print(f"✅ Loaded {len(streams)} snapshots")

        # Process documents
        print("🚀 Starting document processing...")
        pbar = tqdm(desc="Processing docs", unit="docs")

        filter_stats = defaultdict(int)

        try:
            for doc in dataset:
                pbar.update(1)

                # Process document
                kept = self.process_document(doc)

                # Log progress
                if self.processed % 25000 == 0:
                    progress_pct = (self.total_tokens / TARGET_TOKENS * 100) if TARGET_TOKENS > 0 else 0
                    print(f"\n🔎 Processed: {self.processed:,} | Kept: {self.kept:,}")
                    print(f"🎯 Tokens: {self.total_tokens:,} / {TARGET_TOKENS:,} ({progress_pct:.1f}%)")
                    print(
                        f"📊 Top topics: {dict(list(sorted(self.topic_tokens.items(), key=lambda x: x[1], reverse=True))[:3])}")

                # Check if we've reached our target
                if self.total_tokens >= TARGET_TOKENS:
                    print(f"\n🎯 Target reached! {self.total_tokens:,} tokens collected")
                    break

                # No document limit - only stop when target is reached

        except KeyboardInterrupt:
            print("\n⏹️  Processing interrupted by user")
        except Exception as e:
            print(f"\n❌ Error during processing: {e}")
        finally:
            pbar.close()

        # Final upload
        if self.doc_buffer:
            print("\n💾 Uploading final chunk...")
            self.upload_chunk()

        # Final statistics
        print("\n" + "=" * 70)
        print("🏁 PROCESSING COMPLETE")
        print("=" * 70)
        print(f"📊 Documents processed: {self.processed:,}")
        print(f"📊 Documents kept: {self.kept:,}")
        print(f"📊 Total tokens collected: {self.total_tokens:,}")
        print(f"📊 Target was: {TARGET_TOKENS:,} tokens")
        print(f"📊 Achievement: {(self.total_tokens / TARGET_TOKENS * 100):.1f}% of target")
        print(f"📊 Files uploaded: {self.file_count}")
        print(f"📊 Average docs per file: {(self.kept / max(self.file_count, 1)):,.0f}")
        print(f"📊 Retention rate: {(self.kept / self.processed * 100):.2f}%")

        print(f"\n🎯 TOKEN SUMMARY:")
        print(f"   • Collected: {self.total_tokens:,} tokens")
        print(f"   • Target: {TARGET_TOKENS:,} tokens")
        print(f"   • Difference: {(self.total_tokens - TARGET_TOKENS):,} tokens")
        if self.total_tokens >= TARGET_TOKENS:
            print(f"   ✅ TARGET ACHIEVED! ({(self.total_tokens / TARGET_TOKENS * 100):.1f}%)")
        else:
            print(f"   ⚠️  Still need: {(TARGET_TOKENS - self.total_tokens):,} more tokens")

        print(f"\n🎯 TOPIC DIVERSITY ANALYSIS:")
        print(f"   📚 Total unique topics: {len(self.topic_tokens)}")
        print(f"   📊 Topic distribution:")
        for topic, tokens in sorted(self.topic_tokens.items(), key=lambda x: x[1], reverse=True)[:15]:
            percentage = (tokens / self.total_tokens * 100) if self.total_tokens > 0 else 0
            print(f"      {topic:15s}: {tokens:>12,} tokens ({percentage:5.1f}%)")

        print(f"\n📈 Top 10 domains by tokens:")
        for domain, tokens in sorted(self.domain_tokens.items(), key=lambda x: x[1], reverse=True)[:10]:
            percentage = (tokens / self.total_tokens * 100) if self.total_tokens > 0 else 0
            print(f"  {domain}: {tokens:,} tokens ({percentage:.1f}%)")


def main():
    """Main function - update your credentials here"""

    # ⚠️  UPDATE THESE WITH YOUR ACTUAL CREDENTIALS ⚠️
    aws_access_key = AWS_ACCESS_KEY  # Replace with your access key
    aws_secret_key = AWS_SECRET_KEY  # Replace with your secret key
    s3_uri = S3_URI  # Replace with your S3 bucket path

    # Validate configuration
    if aws_access_key == "YOUR_ACCESS_KEY_HERE":
        print("❌ Please update AWS_ACCESS_KEY with your actual access key")
        return

    if aws_secret_key == "YOUR_SECRET_KEY_HERE":
        print("❌ Please update AWS_SECRET_KEY with your actual secret key")
        return

    if s3_uri == "s3://your-bucket/path/":
        print("❌ Please update S3_URI with your actual S3 bucket path")
        return

    # Create and run pipeline
    pipeline = FineWebPipeline(aws_access_key, aws_secret_key, s3_uri)
    pipeline.run()


if __name__ == "__main__":
    main()
